# fix:S1 (17 Nov): the agent's final report

Working paper. Nothing here is a rule.

S1 fix round report. Worktree /home/user/mesozoic-wt/wpc-s1, branch wt/wpc-s1. This is simulation only: everything I ran was plant-level physics in memory, with no record and no episode on a look set.

**Merge**
- The working branch moved during the round, from fccebe42 to a5aac7a8 (the CN, CT and CW follow-ups). I merged it after my two fix commits as merge commit 3f239d42.
- The merge was clean (merge-tree exit 0). It touches no file of S1.
- `git merge-base --is-ancestor` now holds.

**Per finding**

1. **For the lead: the gain-copy split's side effect.** I confirmed it. `control.Tracker` reads `p.kp`/`p.kv` into its ankle cap when it is built (control.py:1240). So a served R2 row whose tracker is built inside the episode takes the run's kv at J+ and W1 (the reviewer saw 11.28 against 2.08). Nothing changed.
   - Options:
     - Keep the split and amend 2d's gain-copies table and the note to ruling 21.
     - Set the copies at the first step. This breaks the delay-queue rule at a mid-task start.
     - Set both the copies and the gains at entry. This breaks G8s's reset.
     - S3 builds the served row's tracker before entry. Ruling 21 calls this "not taken".
   - Recommend: keep the split and amend the text. Only J+ and W1 move, and both are reported only.

2. **X6L's peak was written at bind: fixed.**
   - `ServoEnvelope.restore` now writes the nominal peak while the layer is not acting. `_start` writes `peak_run` at the first physics step.
   - The HOOKS_RULE text is updated. The X6L exception is gone from T4.
   - New test: `test_hooks_the_entry_order_does_not_change_the_reset[X6L, N]`. Whether the layer enters before or after the reset, the forward pass, the first reading and 20 step digests are the same, and the peak is 31.5 from the first step.

3. **Test gaps: fixed.** I added:
   - `test_t7_the_force_range_follows_the_sagged_bus`: V- at a fast state, 3 steps against `line_limits` at the sagged V, with V below 20 V (M08, M36).
   - `test_t10_set_const_inside_the_layer_and_after_the_undo`: `setconst_changes == []` inside, under W1's inertia (M29).
   - `test_t8_the_box_servos_readings_refresh_with_their_targets` (M19).
   - `test_t6_the_inside_play_share_is_the_count_by_hand` (M24).
   - T8's box test now starts from a nonzero command (M31).

4. **A swapped envelope was half-applied: fixed.**
   - `_step` now refuses a physics step when `p.envelope` is not the layer's.
   - `exit` raises after restoring if the envelope or the step was replaced inside.
   - New test: `test_t4_an_envelope_swapped_inside_is_refused_and_restored`. It also checks that the ledger's pattern of swapping and undoing before the next step still works.

5. **The mass guard accepted any changed trunk: fixed.**
   - The factor is now computed as trunk mass over built trunk mass. Built trunk mass = `meta.mass_kg` − the other bodies, which `set_trunk_mass` keeps as built. It must equal `params.mass` to rel 1e-9.
   - A factor of 1 still leaves the env's own draw alone.
   - The trunk inertia is not checked: the plant keeps no built inertia to compare with.
   - The extended test refuses M+ at ×0.9, M- at ×1.1 and M+ at ×1.05, and accepts M+, M-, W1 and W2 at their own factors.

6. **For the lead: the five cells with no kind in the design** (armature neck, kp_scale tail, kv_scale neck and tail, latency_steps). Nothing changed.
   - The code gives all five "the study's assumption", and adds "nominal [A]: ..." to the source text of kp_scale tail and kv_scale neck and tail.
   - Options:
     - (a) Keep as built.
     - (b) Remove the added "[A]" phrases so the source text is only the design's. This changes TABLE_SHA256.
     - (c) The design's author writes the kinds into section 3, and the code follows.
   - Recommend: (c), with (a) until then. The counts stay 1 + 4 + 6 + 28 either way.

7. **For the lead: trap 8 wording.** The code refuses to enter over an existing step hook; the design says it "restores whatever step it found". Nothing changed. Recommend: keep the stricter refusal and reword trap 8.

8. **For S2: `train2.SOURCE_FILES` lacks `servo` (and `g8s`).** It is still absent after the merge (train2.py:141-145). It is not my file. S2 should add both.

**The builder's items (b) and (d), still for the lead**
- (b) The first box-servo target is taken at step 0. It is now tested. The alternative is to hold the reset's target until the first phased step.
- (d) X6L's slope is 6.23, from N's corner and zero, against the X6-60's 8.30. The alternative is a line of its own from the X6-50-L's figures. Recommend: keep as built while X6L is reported only.

**Checks that fail without each fix**
- HEAD's old `servo.py` against the new tests: `4 failed, 54 passed in 3.75s`. The four are test_t4_exact_undo[X6L], test_a_mass_setting_needs_the_callers_trunk_mass, test_t4_an_envelope_swapped_inside_is_refused_and_restored and test_hooks_the_entry_order_does_not_change_the_reset[X6L].
- The reviewer's 36 mutants, regenerated from the fixed file:
  - The unchanged copy: `58 passed in 2.11s`.
  - M08 `1 failed, 57 passed in 2.08s`; M19 `1 failed, 57 passed in 2.11s`; M24 `1 failed, 57 passed in 2.19s`; M29 `1 failed, 57 passed in 1.98s`; M31 `2 failed, 56 passed in 5.18s`; M36 `1 failed, 57 passed in 2.81s`.
  - Every one of the 36 fails at least one test.

**Final outputs (after the merge)**
- Servo tests: `58 passed in 2.34s`. The state of the first fix commit gave `55 passed in 2.60s`.
- Hashes: `45719d030359ec7d5d3914af3814d837fdc05d77652ba45c64792b511027528d c0027992de21067525cc4489b74acb157feb89f3a4c7618580addc2b341be1c1`, the registered pair.
- TABLE_SHA256 is unchanged: 342f3948cef116ea7fde49106817596a3d69ffa8d85c84dcb7c35f442546ef57.
- ruff on both files: `All checks passed!`.
- Acceptance probe, re-run on the fixed layer:
  - "exact undo yes" for all 18 settings.
  - "every physics step's digest as before: True".
  - Baseline 89.8 µs per physics step. The layer's own arithmetic was 5.9 µs/step for today and 39.8 µs/step for N.
  - Load average was 3.6, so the timings are noisy.

**Not done**
- The full test suite. The merged files are byte-identical to the working branch's.
- The S2 to S4 items.
- No independent review of this fix round.
- The main checkout shows three untracked review files under stage2/design/checkpoint/reviews/ (roundC__build_CF/CG/FU_17nov.md). They are not mine.

**Files**
- My scripts and outputs are in /tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/wave6/s1/mut_fix/ (results.txt, results_key.txt, run_all.sh).
- The new probe output is in /tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/wave6/s1/probe_fix/.
- Both folders are outside the repository.

`git diff claude/mesozoic-trex-stage2-refine-4jffre --stat`:
```
 hardware/trex_form_study/designlab/stage2/servo.py | 1334 ++++++++++++++++++++
 .../trex_form_study/tests/test_stage2_servo.py     |  906 +++++++++++++
 2 files changed, 2240 insertions(+)
```
`git status --porcelain`: empty.

**Commits (oldest first)**
- b68648d2 and 6afd4f21 (the builder's)
- 689f1df3 Servo model S1 fix: X6L's peak at the first physics step, a swapped envelope refused, the trunk at the setting's factor
- 0d74ad43 Servo model S1 tests: the gaps the review's mutants found
- 3f239d42 Merge the working branch into S1's fix round

Branch head: 3f239d421a8c1b6c118ad0019c736216fe9bc1e0. Nothing pushed.
