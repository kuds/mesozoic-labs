# review of fix:CE (roundC): the independent reviewer's final report, 16 Nov 2026

Working paper. Nothing here is a rule.

VERDICT: merge as it stands. All nine findings are fixed in the way the rulings say, each with a test that fails when the fix is reverted. I found no blocker and no major. Eight minor points follow; 1 to 3 touch CE's source files, so they are best done before any run fixes the source hashes.

Reviewed: wt/wpc-ce at 16890e8 against 8d8c450 and against the working branch tip 373903a. My scripts and logs are in /tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/review_ce_fix/. No tracked file changed, nothing committed, and no file in either checkout is newer than my first scratch file.

FINDINGS (all minor)

1. designlab/stage2/r5env.py:278-279 (`_check_freeze`). When the named freeze is the set's latest, the env returns without comparing the task table with the manifest.
   - Evidence (k4_same_freeze.py): a synthetic set frozen once with nominal, v0.4, start and nominal_to_v0.4. I then set `available: false` on nominal_to_v0.4 in transitions.json, with no new freeze. Same settings, tasks=(): "ACCEPTED; tasks ['stand', 'walk_nominal'] ... started/now 0 0". Before the edit the tasks were stand, walk_nominal, walk_v0.4.
   - This breaks FREEZE_RULE's own definition (a task is in the table exactly when the manifest holds every file of its plan). It needs an edit of a frozen set's records outside the process.
   - Fix: run part (2) in the equal-hash case too, with `setio.frozen(lib.set_doc)["entries"]` as the manifest.

2. designlab/stage2/r5lib.py:595-598. No freeze covers set.json's `crouch_frac`, and the library does not compare it with the frozen walk entries' own value, as `bar.set_crouch(set_doc, refs, walks)` does.
   - Evidence (k4_cases.py, case C4): set.json edited 0.03 to 0.08, entries still 0.03, set frozen again. Result: "C4 ACCEPTED ... omega 4.828071509019948 -> 4.97878143077127".
   - Effect: only the critic's DCM-error inputs move; the physics does not.
   - This predates the fix round; it surfaced in the K4 probe.
   - Fix: refuse a set whose set.json crouch differs from a used walk entry's.

3. designlab/stage2/r5core.py:140-201 (`REWARD_RULE`). Ruling 1 asks for exactly what the code computes. Every formula matches; these small points are unstated:
   - a. In an episode's first env step the displacement behind tracking's speed and yaw rate and behind progress's sample spans 18 ms and is divided by 20 ms (r5core.py:470, 716-720, 804). Evidence (first_step2.py): at the reset the CoM read equals the reset pose's (difference 0.0e+00); after env step 1 it equals the pose after 9 physics steps (0.0e+00, against 7.7e-04 for the pose at the step's end). It matters only for a start in motion.
   - b. "every penalty 0": progress reads up to 2e-05 on a playback of V0's walks (the tests allow below 1e-4).
   - c. duty's n is clamped to 1..256 env steps (line 733).
   - d. clearance (a) is skipped when T <= 0 (line 757).
   - e. The "true state" sentence names the last physics step only. The per-physics-step readings (contact, flight, impact, touchdown, the duty shares, the servo term's actuator forces) are each taken as of that physics step's start.
   - f. `HEAD_RULE`'s first sentence is R5's. In R6 the overlay is added to the settled home pose (`Core.targets`); the settled jaw is 0.0015 rad on 75-M-H-10-X, so an R6 jaw target reaches 0.4515 rad.
   - Fix: one sentence each, a and e in the wording file's entry 1a as well.

4. stage2/design/CE_RULE_WORDING.md has no entry for T3. Commit 225f4b3 changed what a mid-task start does: at k = 0 the servos are set to the command of one physics step before the start, a forward pass follows, and the estimator is reset a second time from that pass. `MID_START_RULE` says so; T3 in the draft does not. Fix: add an entry, Label B.

5. stage2/design/CE_RULE_WORDING.md:50-51 (entry 1b). The new W1 sentence says "the tracking term at least 0.95" without scope. It was measured on Stage 1's arrays and V0 only (smallest 0.951, Stage 1's turn in place); V0 has no 1.0 rad/s turn and no turn in place. `REWARD_RULE` itself says "close to 1". Fix: scope the number to those two sets, and write "every penalty below 1e-4" (3b).

6. designlab/stage2/r5lib.py:771 (`from_plan`). It cannot name the heading "reference". A row that plays a turn-in-place plan as a plan gets rule E4's integral, while the policy trained with a heading input near zero there.
   - Nothing runs this today: the bar defines no turn-in-place condition and V0 has none. Entry 5a states it.
   - For the lead: add a `heading` argument before Track C defines that reading.

7. designlab/stage2/r5env.py:585-605 (`describe`), for whoever builds the trainer:
   - Across an accepted later freeze three description keys change: `set_freeze`, `set_freeze_now` and `inputs.freeze` (case C6). `set_freeze` is the latest freeze's hash, not the settings'. A trainer that compares the description whole would refuse what K4 accepts.
   - Two freezes with the same manifest: `set_freeze_started.index` is the later index, not the one the run started on (cases C3, C10).
   - `env_configs()` names no tasks, so any later freeze that adds a usable task refuses the resume. That is K4 to the letter and entry 8 flags it. Storing the resolved task names in the configs would make K4's accepted case reachable.
   - A named task that left the table raises ValueError before the freeze check, not `tune.InputError`.

8. tests/test_stage2_r5env.py:
   - :451 `test_a_mid_start_draw_at_the_end_of_the_last_span_does_not_land_in_the_hold` also passes with the old inline logic; only the unit test at :424 pins the fix.
   - The branch "was in the table under freeze i and is absent now" has no test. It works: my k4_leaves.py got the refusal naming walk_v0.4 and the withdrawn file.
   - The fixture `freeze_t0` (:763) lists every entry in the manifest whether available or not; `sets.manifest_of` lists available entries only.

THE NINE FINDINGS, EACH CHECKED BY REVERTING THE FIX IN MEMORY (a pytest plugin patching the imported modules; reverts.log)

| Finding | Ruling met | Tests failing on revert |
|---|---|---|
| 1 Blocker, the readings | Rule strings rewritten; four new tests pin duty, progress, clearance (a) and the instants | duty condition removed: 8; progress to the literal form: 5 |
| 2 Turn in place | `TaskDef.heading = "reference"` for inplace_left and inplace_right only; walking turns and `from_plan` keep E4 | 3 |
| 3 Later freeze | `_check_freeze` accepts and refuses as K4 says | 2 |
| 4 Jaw | Jaw 0.3-0.45 rad, gaze +-0.45 rad, `Core` refuses a body without the margin | 5 |
| 5 Estimator's first reading | Fixed for k = 0 and k > 0 | 3 (2 with only the second reset removed) |
| 6 0.30 m/s band | Reads `Script.target_speed` | 1 |
| 7 Rounding in the mid-start draw | `mid_start_time` | 1 (unit test) |
| 8 Unstated numbers | Entries 7a-7c | not applicable |
| 9 X1 docstring | Both dates, pinned by a test | not applicable |

WHAT I CHECKED AND FOUND RIGHT

- Rule strings: I compared `REWARD_RULE`, `READINGS`, `HEAD_RULE`, `COMMAND_RULE`, `MID_START_RULE`, `RESET_ORDER`, `DRAW_ORDER` and `FREEZE_RULE` with the code term by term. No mismatch beyond finding 3.
- Kinematic playbacks of all 15 of V0's task scripts (no physics step; v0_playback.log) reproduce the docstring's and entry 1b's numbers:
  - heading maximum 0.351, 0.1395 and 0.0447, on 98/99, 148/149 and 207/208 steps, means 0.021, 0.013, 0.006;
  - tracking minimum 0.9891 to 0.9938 on the walks and 0.9618 in the quick turn;
  - every other penalty 0;
  - V0's v0.5 entry holds 0.49999999999999994 and v0.3 holds 0.3.
- Head script in closed loop (S1 view, zero action, both -X bodies, 32 episodes; head_dyn.log): at nominal, at kp 0.6 and 1.2 with 24 ms, and at low damping, the range term is 0.000 throughout. The smallest margin is 0.1424 rad on the neck yaw and 0.1467 rad on the jaw. Model and usable ranges are +-0.6 and 0 to 0.6 on both bodies.
- K4 (k4_cases.py): an unused entry changed is accepted; a tampered old manifest is refused; a task entering or leaving is refused with the task and file named; a file changed and changed back is accepted. The real set V0 builds with its freeze named, and an unknown hash is refused with the message naming the latest.
- What a resumed run compares: the named freeze's manifest as set.json holds it (checked against the hash in the settings), against the files read under the latest freeze, for the env's tasks plus the nominal walk, and the table by manifest membership. It compares nothing the run itself stored.
- `mid_start_time` (midtime_fuzz.py): "checked 359581 outside any span: 0 offset mismatches: 0". This covers u = 0, u = total, one ulp either side, every cumulative span sum, and contiguous and gapped spans. A span's end is bit-equal to the next segment's t0 (control.py:1028), so the last instant before it is never in the hold.
- Mid-start reading: in R6 and on carpet the estimator's and the observation's first loads equal the pass on the zero action's command to 6e-08 body weights. Over 60 mid starts per delay no foot reads over 2 body weights (was 60 % of resets); the largest is 1.25 at k = 0 and 1.73 at k = 5.
- Side effects: nothing in the working branch or in any other wt/ branch imports the four modules. The draw order is unchanged. The g8a fixture's `stand_in_walks` and the reset added to `test_factors_are_applied_and_restored` are test-only.
- Record-key hashes, printed in the CE worktree: 45719d030359ec7d5d3914af3814d837fdc05d77652ba45c64792b511027528d c0027992de21067525cc4489b74acb157feb89f3a4c7618580addc2b341be1c1.
- Wording file: all 18 "Old" quotes are verbatim in the draft (quotes.py). Every New sentence matches the code and every label is truthful, with commits 0dc39b8, ad64656, d1e667c and ebae071 as named. Entry 9's 10,539 steps is right for walk/v0.4.
- G8a stock form after the fix round: 75-M-H-10-X walk/brisk seed 3043 k 7, and 60-N-T-10-X walk/nominal seed 3043 k 3, both identical over 3000 of 3000 steps.
- Merge: `git merge-tree --write-tree` exits 0 with no conflict. The merge base is the working tip itself (0 behind, 25 ahead). The result differs from the tip by exactly the nine files, all added. CE's own non-merge commits touch only those nine.
- ruff check passes on the eight files.

TESTS (exact summary lines)

- tests/test_stage2_r5lib.py tests/test_stage2_r5core.py: "56 passed in 31.34s"
- tests/test_stage2_r5env.py tests/test_stage2_g8a.py: "36 passed, 1 skipped in 58.17s" (the skip: tests/test_stage2_r5env.py:496, "slow: set STAGE2_SLOW=1")
- tests/test_stage2_tbenv.py: "11 passed in 0.57s"

BAR FORM

`g8a.bar_form("75-M-H-10", "V0", ("walk/nominal",), (3043,), (0, 5), root=<main checkout>/stage2/sets)`:
- document: "identical": 2, "of": 2, "not_run": [], "passes": true; set_freeze index 0, 34a820e5...
- R0@0ms: "reference_steps": 7030, "env_physics_steps": 7030, "compared_steps": 7030, "first_difference": null, "reference_fell_step": null, "env_terminated_step": null, "ends_alike": true, "identical": true
- R0@10ms: the same fields with the same values, "identical": true

NOT CHECKED

- The slow test (STAGE2_SLOW=1) and the rest of the suite beyond the three groups.
- Any training run, policy or look set.
- The turn in place and the 1.0 rad/s turn on a real set (V0 has neither; turn in place only on Stage 1's arrays).
- K-stride windows on a real set (V0's are all one stride).
- Throughput.
- Entry 7a's claim that the 4 s hold covers the bar's stop item.
- The 64.3 degrees for 60-N-T-10 in entry 5a.
- STAGE2_PLAN.md itself; I read the Track B draft's rules only.
- A second bar_form on V0, as you asked for one.
