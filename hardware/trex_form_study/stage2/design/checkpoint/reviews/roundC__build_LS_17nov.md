WP L builder report: rulings 18, 23 and 26 (the code part). Worktree /home/user/mesozoic-wt/wp-l, branch wt/wp-l. Everything here is simulation only (prescribed motion). I froze nothing, pushed nothing, rewrote no history and committed no look-set record.

**1. Merges**
- 8555f72f: the working branch at 9889291c. Clean, no conflicts. Before this merge, 036ec9d6 differed from wt/wp-l only by the fixer's review file.
- 38842ac2: a second merge, also clean. The working branch moved while I was building (32b5ecc5: rulings 28–35, the merged LB16, CG2 and CN3 work). It touches no file of this package.
- I ran all tests, verify and hashes on 38842ac2.

**2. Per item**

(1) **Ruling 18** (447acad1). In `designlab/stage2/moves_look.py` I added:
- the rule `S2L-L1L4-start` in `RULES`, and `START_RULE_ID`;
- `mixes_stage1_start` (true for a start from the stand into a walk; false into a turn gait or the turn in place);
- `rule_ids`, which now appends the rule for such a start;
- `stage1_start_law`, `start_mix`, `_start_duty` and `start_mix_record`;
- a `Bind.start_law` field. `laws_for`'s `lift` mixes the lift peak at the swing's lift-off. `bind_move` mixes the duty into the start's schedule function. `entry_extra` writes `look.start_mix`.

The mix is exactly Stage 1's value at amp 0 and exactly the look walk's at amp 1.

Where Stage 1's start law comes from: `moves.stage1(body, "start", H)`, which is `transitions.TRANSITIONS["start"]`, the stand to "nominal". `refs.walk_specs` maps that to `sizing.NOMINAL`.
- **Duty 0.60:** NOMINAL has `swing_time` None, so its duty is `spec.duty`, the constant duty of `transitions._param_fns("start")`.
- **Lift peak 0.5:** Stage 1's laws are `moves.TODAY`, whose lift sin²(πu) is `gait.lift_profile(u, 0.5)`, the GaitSpec default `lift_peak`.
- These are the same on every body. The builder's 0.5 was right.
- The lift height is not mixed. It is 0.10 H on both Stage 1's nominal and the look nominal.

The mixed swings, the same on every body (the later swings are the look walk's: duty 0.686465, peak 0.375):

| Swing | amp | duty | lift peak |
|---|---|---|---|
| left, lift-off 0 | 0.05792 | 0.605008 | 0.49276 |
| right, lift-off 0.5 | 0.68256 | 0.659018 | 0.41468 |

Tests, in tests/test_stage2_moves_look.py:
- `test_l1_l4_in_a_start_mix_stage1s_start_in_by_the_amplitude_ramp[nominal|v0.4]` checks the arrays and the entry. On the old moves_look it fails: "assert 0.6864653129237532 == 0.6050080709245438 ± 1.0e-12" (nominal) and "0.5862816369688507 == 0.5992054324132358" (v0.4).
- `test_the_start_mix_is_no_other_moves_law` covers a change (ramp 1 and 2), stop2, stop1, a turn and the in-place start. Their plan arrays equal, bit for bit, the plans made with the mix switched off. On old code it fails with an AttributeError.
- `test_rules_are_recorded` is updated.
- In the scratch build, `nominal_to_v0.4` rebuilt with the new code hashes equal to the committed entry on all three 75 cm pairs: 19fd45aa…, 5441580…, a4673b58….

(2) **Ruling 23** (c8ad130a). In `designlab/stage2/sets.py`:
- New: `STOP_GROUPS`, `blocked_list`, `blocks_in_records`, `_blocked_after`.
- Changed: `merge`, `build` (an in-run `stopped` list; `res["blocked"]` is the record's list or None), `verify` (one "the set is stopped" line per group), `status` and `main`.
- `blocked` is a list of {by, reason} in start, nominal_to_v0.4 order, or null when nothing stops the set.
- A dict or null `blocked` in an older record still reads. The frozen sets' records stay null (tested).

Tests:
- `test_blocked_names_every_failing_group_and_a_passing_merge_lifts_only_its_own` (test_stage2_sets.py, stand-in set).
- `test_the_frozen_sets_records_name_no_stop[V0|F25F9|F25F9cl]`.
- `test_a_passing_merge_does_not_lift_another_groups_stop_on_a_look_sets_records` (test_stage2_moves_look.py): the fixer's demo on a tmp copy of V6/60-N-T-10's records. On old code it fails with "TypeError: 'NoneType' object is not iterable", because the stop was lifted.
- Dict-shape tests adjusted:
  - `test_a_failed_start_stops_the_set`: old lines 713 and 716.
  - `test_a_built_sets_lead_in_follows_its_walks`: old lines 2242 and 2250. These branches are not reached on the committed round sets.

(3) **Ruling 26** (f8f56820). In `sets.py` `_produce_turns`, on a look set each direction's reason names only its own failed gates. The rate's `no_reference` gives each direction's reason where they differ. The round's sets keep their old wording.
- Test: `test_a_look_sets_turn_reasons_name_only_the_gates_each_direction_failed` (test_stage2_sets.py, new fixture `look_turns`).
- On old code the right turn's reason ends "; also outside join_command", so the test fails.

**Extra** (403987c6): on a look set, `merge` adds any `moves_look.RULES` text missing from transitions.json `rules`, so an existing set gains the S2L-L1L4-start text. Test: `test_a_merge_into_a_look_set_built_before_a_rule_records_its_text`; old code raises KeyError 'S2L-L1L4-start'.

**3. Scratch read (not a record)**

I built into /tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/lstart/scratch_sets/, from copies of the committed records and walks. The numbers are in /tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/lstart/read_start.json.

**Every start now passes and is within the Stage 0 limits.** On every one, every entry gate passed. The knee limit is 7.558 rad/s on all bodies, and ratios are against it.

| Set/body | Knee ratio (before) | Peak: knee, phase, swing u | join_a com/zmp (mm) | join_b com/zmp (mm) | zmp_error last (mm) | Duration |
|---|---|---|---|---|---|---|
| V6/75-M-H-10 | 0.8041 (1.1261) | left, 1.05, u 0.159 | 0.509 / 0.232 | 0.275 / 0.530 | 1.234 | 5.661 s, 800 frames |
| V6noF5/75-M-H-10 | 0.8453 (1.1261) | right, 2.55, u 0.159 | 0.509 / 0.232 | 0.263 / 0.511 | 1.224 | 5.661 s |
| V6/75-M-H-10-X | 0.7948 (1.1245) | left, 1.05 | 0.512 / 0.233 | 0.274 / 0.537 | 1.247 | 5.661 s |
| V6/60-N-T-10 | 0.9216 (1.2340) | left, 1.05 | 0.408 / 0.219 | 0.354 / 0.523 | 1.266 | 5.093 s |
| V6noF5/60-N-T-10 | 0.9791 (1.2340) | right, 2.55 | 0.408 / 0.219 | 0.349 / 0.513 | 1.266 | 5.093 s |
| V6/60-N-T-10-X | 0.9025 (1.2232) | left, 1.05 | 0.412 / 0.221 | 0.352 / 0.503 | 1.246 | 5.093 s |

- The gates are ik_ok (0 failed frames), join_a, join_b and zmp_error (limit 4 mm), plus rolling_split (56 rolling frames, 0 failed) on the sets with F5. V6noF5 has no rolling frames and no rolling_split gate.
- **The knee peak has moved.** It now falls in a swing after the ramp (amp 1, the look walk's own swing), no longer in the first swing.
- V6noF5/60-N-T-10 is close to the limit: 0.9791, against its own nominal walk's 0.9814.
- V6/75-M-H-10-X is governed by neck peak 0.8052, not the knee.
- `nominal_to_v0.4`, 75 cm only: V6 0.9188, V6noF5 0.9192, -X 0.9044 (right knee, phase 0.55625). All pass, with arrays bit-equal to the committed ones.
- The 60 cm copies stay blocked by `nominal_to_v0.4`, from the old record. I did not rebuild those changes.

**4. Outputs**
- Tests on 38842ac2:
  - moves_look + moves + sets: "254 passed, 3 skipped in 371.86s (0:06:11)". I did not list the 3 skip reasons in this run.
  - lookbar, whole: "44 passed, 6 skipped in 62.04s (0:01:02)". The 6 skips are STAGE2_SLOW whole episodes. The state-A tests `test_the_modules_of_every_existing_record_key_are_state_as` and `test_no_round_a_or_stage_1_file_differs_from_state_a` passed.
- Verify, the same before and after (the full logs, notes included, are identical):
  - "V0 75-M-H-10: verify ok: 20 references, 2 not available, 0 failures; freeze 0"
  - "F25F9 75-M-H-10: verify ok: 12 references, 11 not available, 0 failures; freeze 0"
  - "F25F9cl 75-M-H-10: verify ok: 20 references, 2 not available, 0 failures; freeze 0"
- Hashes: `45719d030359ec7d5d3914af3814d837fdc05d77652ba45c64792b511027528d c0027992de21067525cc4489b74acb157feb89f3a4c7618580addc2b341be1c1`

**5. Diff and status**
`git diff claude/mesozoic-trex-stage2-refine-4jffre --stat`:
- moves_look.py +113
- sets.py 111
- tests/test_stage2_moves_look.py +170
- tests/test_stage2_sets.py +130
- "4 files changed, 492 insertions(+), 32 deletions(-)"

`git status --porcelain` is empty. All outputs are outside the repo, so `git check-ignore` does not apply. Nothing new was written in the worktree.

**6. Commits, oldest first:** 8555f72f, 447acad1, c8ad130a, f8f56820, 403987c6, 38842ac2. Head: 38842ac2.

**7. For the lead**
- **Turn in place:** its start (`hold_move`) is not mixed, under its own law L13 and "no other move". The alternative is to mix it too, which would also move its periodic cut.
- **Duty readers:** the duty is mixed in the start's schedule. In the first stride, L5's bob trough and L6's roll phase therefore read the mixed duty, as in a change. The alternative is to keep them on the look duty.
- **Lead-in start:** `start_v0.4` also gets the mix, against Stage 1's start into nominal. Stage 1 has no start into v0.4. Its lift height (0.11 H) is not mixed.
- **Single-process builds:** a failed start still stops the run, so a `--procs 0` or `--procs 1` build names only the start. With 2 or more processes both groups run. Making both stop groups always build changes the counts in the state-A test at old line 714.
- **Duplicated stop rule:** `blocks_in_records` restates `produce`'s stop rule from the records. I kept it because the older dict records (V6/60, V6noF5/60, V6/60-X) had lost the start's stop.
- **Turn reasons:** per direction on the look sets only. Making this universal would change the state-A test `test_turns_a_leg_drive_over_its_limit…`.
- **Extra commit:** 403987c6 and the `look.start_mix` key go beyond the three items. Drop them if you prefer.
- **Not done:**
  - ruling 28 (the causal preview grid), ruling 30's review citation and ruling 32(A), all from the newly merged texts;
  - any rebuild or commit of look-set records;
  - the 60 cm `nominal_to_v0.4` rebuilds;
  - `moves legacy` and `regress check`.
