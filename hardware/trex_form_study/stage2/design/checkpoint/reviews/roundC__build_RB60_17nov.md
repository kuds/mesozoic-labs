WP L builder report: the three 60 cm look sets rebuilt. Worktree /home/user/mesozoic-wt/wp-l, branch wt/wp-l, head 8bc47346.

Everything was prescribed motion in simulation. Nothing was frozen or pushed, no history was rewritten, and no episode, panel, fit or training ran. moves.py, bar.py, the code files and the 75 cm records have no diff since a3ea12c0.

**Headline.** All three 60 cm sets are stopped at `nominal_to_v0.4`, as before, so none could be frozen:
- **Start:** passes its gates and Stage 0 on every set (ruling 18). Before, it failed at knee 1.2340.
- **nominal_to_v0.4:** "no transition within the gates" (A8), over the Stage 0 knee-speed limit. Every gate of its entry passes.
  - V6: 1.0958 / 1.0678 (one-stride / two-stride ramp)
  - V6noF5: 1.0964 / 1.0679
  - -X: 1.0609 / 1.0337
  - Both tries are bit-equal to the previous build on every set.
- **Not built:** the stops and turns, because the set is stopped.
- **Turn in place:** built; not available on every set.

**1. Merge**
- a61c53d2 merges the working branch at cab73e8b (LT2 to LT4, servo S2, rulings 37 to 40, the RB75 build report). There were no conflicts; its tree equals the `merge-tree` result, 0781f609.
- All eight module hashes recorded in the 75 cm entries equal the merged HEAD's. So the 60 cm sets were built by the same code as the 75 cm sets.
- The working branch moved again during my builds (tip c795ce1b: the 75 cm cherry-picks, P42, servo S3). I did not merge it again.
  - `git merge-tree --write-tree HEAD claude/mesozoic-trex-stage2-refine-4jffre` is clean.
  - Against that tip, the only package files that differ are my 60 cm records and V6/tail_phase.json.

**2. Code points.** The 75 cm check listed no required fix. Its finding 2 was "Proposed (optional)". **I made no code change.**
- I wrote finding 2 (record the dt that `causal_grid` actually passed, and have `preview_grid_record` / `entry_extra` refuse a causal move whose controller did not run on that grid) and then reverted it.
- Its test failed on the old code: "1 failed, 48 deselected in 0.96s", "Failed: DID NOT RAISE LookError". It passed on the new code together with the adjusted `built` stop test: "3 passed, 46 deselected".
- **Why I did not keep it:** it changes moves_look.py's hash. The 60 cm sets would then not be built by the 75 cm sets' code. Every 75 cm entry would also be noted "built by other code", and their not-available entries would no longer count as current.
- **For the lead:** take it in the next package that rebuilds the look sets (for example P42's), or leave it. The patch is saved at /tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/rebuild60/finding2_preview_grid_passed.patch.

**3. Rebuild**

Before the sets, `sets tail-phase --body 60-N-T-10` ran on a clean tree (653 s) and was committed as ebf2a85d.
- All four values and check blocks are bit-equal: v0.3 −1.736603, v0.35 −1.771509, v0.45 and v0.5 −1.780236 rad.
- Only the provenance changed (dirty false). The 75-M-H-10 entries are unchanged.

Each set was built in an empty directory in two steps: (a) `sets build`; (b) the turn in place (`--only inplace_*`), `setscreen run`, `verify` and `status`.
- Every walk is bit-equal to the previous build.
- Every entry has dirty false and one commit per set.
- set.json `rules` names the review as `S2L-join-review` (path and sha256 b3ff4576…). `look.join_review` holds the same.
- `look.cut` is null and `cut3_probe` is null; the cut reason says the review probed no cut-3 leave point on 60-N-T-10.
- `joins_failed` is {}, `freezes` is [], and `blocked` is `[nominal_to_v0.4]`.

Join numbers below are CoM / ZMP in mm. A ratio is the share of the Stage 0 limit.

| | V6/60-N-T-10 | V6noF5/60-N-T-10 | V6/60-N-T-10-X |
|---|---|---|---|
| Walks | All frozen except v0.5 (dropped, knee 1.0163) | All frozen except v0.5 (dropped, knee 1.0406) | All six frozen (v0.5 knee 0.9954) |
| Start | Available. Knee 0.9216, left_knee, phase 1.05. join_a 0.408/0.219, join_b 0.354/0.523 | Available. Knee 0.9791, right_knee, phase 2.55. join_a 0.408/0.219, join_b 0.349/0.513 | Available. Knee 0.9025, left_knee, phase 1.05. join_a 0.412/0.221, join_b 0.352/0.503 |
| nominal_to_v0.4, one-stride ramp | Knee 1.0958, right_knee, phase 0.556. join_a 0.074/0.475, join_b 0.233/0.882 | 1.0964, right_knee, phase 0.556. 0.073/0.498, 0.240/0.886 | 1.0609, right_knee, phase 0.556. 0.074/0.470, 0.233/0.884 |
| nominal_to_v0.4, two-stride ramp | Knee 1.0678, left_knee, phase 1.056. 0.068/0.475, 0.226/0.893 | 1.0679, left_knee, phase 1.056. 0.067/0.498, 0.234/0.669 | 1.0337, left_knee, phase 1.056. 0.068/0.470, 0.226/0.858 |
| Other changes | Over knee: nominal_to_v0.35 (1.0668/1.0415), v0.45 (1.1303/1.0981), v0.3 (1.0361/1.0171). Reached instead by v0.4_to_v0.35 (0.9269), v0.4_to_v0.45 (0.9561), v0.35_to_v0.3 (0.9265), all available | Over knee: nominal_to_v0.3 (1.0365/1.0172), reached by v0.35_to_v0.3 (0.9560). nominal_to_v0.35 and v0.45 not built | nominal_to_v0.3 available with the two-stride ramp (0.9913; one-stride ramp 1.0096). nominal_to_v0.35, v0.45, v0.5 not built |
| Stops, turns (`join_command`, `join_b`) | Not built (set stopped) | Not built | Not built |
| Turn in place | Periodic. Thigh twist 0.7988/0.7989 (limit 0.87). Hip-yaw margin 0.0604/0.0603 rad (limit 0.05). The reference is within Stage 0. Not available: its start is over knee speed at 1.1996/1.2566 | Identical to V6 | Periodic. Twist 0.8127/0.8128, margin 0.0562 rad. Not available: start knee 1.1944/1.2546 |
| verify | 9 references, 11 not available, 12 failures (was 5/3/18) | 7/9/14 (was 5/3/18) | 8/7/14 (was 6/2/18) |
| Wall time, s (build + in place + screen) | 3530 + 567 + 205 | 3523 + 560 + 153 | 3057 + 616 + 213 |

**nominal_to_v0.4, which swing.** I rebuilt both tries into scratch; all 6 sha256 equal the records' `built_sha256`. The peak is the first swing that lifts from L10's flat stance:
- **One-stride ramp:** the right swing lifting at phase 0.5, with foot pitch 0 at lift-off. The left swing at 1.0 is next highest: 1.0674 (V6), 1.0676 (V6noF5), 1.0336 (-X).
- **Two-stride ramp:** the left swing lifting at 1.0. The right swing at 1.5 reads 1.0547, 1.0548 and 1.0225.
- **Earlier swings:** with V6's toe-off they lift with a pitch of 0.299 rad and read 0.66–0.77. Once the toe-off returns (0.0305 rad), the v0.4 walk's swings read about 0.88 on V6 and 0.86 on -X.
- **V6noF5:** all its stances are flat, and its nominal swings already read 0.97–0.98.
- This reproduces fix_L item 10's diagnosis.

**Freeze.** None of the three can be frozen:
- **The stop:** each is stopped by `nominal_to_v0.4` (A8/A12), and verify fails on the stop and on the 11 or 13 items not built. Under ruling 26, the change to 0.40 m/s goes to the owner with these numbers.
- **The rest it would wait for:** no refcheck.json, ruling 40's freeze text in sets.py, and rulings 29/30 (the lead's reading).
- **P42 (5.0 mm):** decided while these builds ran. It changes nothing here, because no causal move was built on any 60 cm set.

**4. Outputs**
- Tests, the three package files on the final tree: "262 passed, 5 skipped in 470.33s (0:07:50)".
  - Three skips are STAGE2_SLOW.
  - Two are new and by design: test_stage2_moves_look.py:1243 "V6/60-N-T-10's records are no longer the ones the fixer's demo read (rebuilt)", and :1279 "V6/60-N-T-10's records already state S2L-L1L4-start (rebuilt)".
- `tests/test_stage2_lookbar.py -k state_a`: "2 passed, 58 deselected in 1.72s".
- Frozen sets verify:
  - "V0 75-M-H-10: verify ok: 20 references, 2 not available, 0 failures; freeze 0"
  - "F25F9 75-M-H-10: verify ok: 12 references, 11 not available, 0 failures; freeze 0"
  - "F25F9cl 75-M-H-10: verify ok: 20 references, 2 not available, 0 failures; freeze 0"
- 75 cm look sets, unchanged: "V6 75-M-H-10: verify ok: 15 references, 13 not available, 0 failures", "V6noF5 75-M-H-10: verify ok: 17 references, 11 not available, 0 failures", "V6 75-M-H-10-X: verify ok: 15 references, 13 not available, 0 failures". Each status is "not frozen; lead-in start".
- 60 cm verify, exactly as printed:
  - "V6 60-N-T-10: verify FAILED: 9 references, 11 not available, 12 failures"
  - "V6noF5 60-N-T-10: verify FAILED: 7 references, 9 not available, 14 failures"
  - "V6 60-N-T-10-X: verify FAILED: 8 references, 7 not available, 14 failures"
- 60 cm status: "not frozen; missing 11 (V6) / 13; lead-in start; stopped: nominal_to_v0.4: no transition within the gates …".
- Hashes: `45719d030359ec7d5d3914af3814d837fdc05d77652ba45c64792b511027528d c0027992de21067525cc4489b74acb157feb89f3a4c7618580addc2b341be1c1`, the registered pair.
- The arrays (.npz), logs/ and partial/ are git-ignored (stage2/sets/.gitignore lines 1, 6, 4); the records are committed.

**5. Diff and status**
- `git diff claude/mesozoic-trex-stage2-refine-4jffre --stat`: "38 files changed, 37219 insertions(+), 12186 deletions(-)".
  - This package's 16 files: 37107 insertions, 9239 deletions. These are the 15 records (five per set) and V6/tail_phase.json.
  - The other 22 files are the working branch's newer commits that I did not merge (P42 texts, servo S3, gate1.py, the RB75 check report and others).
- `git status --porcelain` is empty in the worktree and in the main checkout.

**6. Commits, oldest first:** a61c53d2 (merge), ebf2a85d (tail phases), 5cff3e9f (V6/60), f2c26b22 (V6noF5/60), 8bc47346 (V6/60-X). Head: 8bc47346. Each message ends with the two attribution lines.

**For the lead**
- **First V6/60 build was killed:** it hit the tool's 30-minute background limit at about 13:34, during the moves. I deleted the directory and built again from empty with a 2-hour limit. The records come from the second build only; that build's log is build_V6_60.log and the first build's is build_V6_60_killed.log.
- **Load rule:**
  - 13:09–13:15: both of my workers ran while the 1-minute load was 2.71–3.78, because another agent's pytest had started. I then paused one by hand (SIGSTOP).
  - After that a script paused and resumed my second worker (governor.log: 15 pauses). It allowed two only when the load was ≤ 1.4, so the load sometimes stood at 2.53–2.80 for up to about a minute before a pause took effect.
  - Pausing changes no result. It does inflate the per-item seconds in the logs (for example "1547 s"); the records hold only `built_by.at`.
- **Which other changes a stopped set holds depends on timing.** `sets.build` cancels only the futures not yet handed to a worker, so the items in flight when the stop arrives still get built. V6 holds nominal_to_v0.35, v0.45 and v0.3; V6noF5 and -X hold only nominal_to_v0.3.
  - Options: leave it, since the set is stopped anyway; build the rest with `--only`; or make the build drain or refuse deterministically.
- **Two V6/60 demo tests now skip** because they read the old records. Option: have them read 2a5466ab's records from git.
- **Turns and stop:** not built, because the set is stopped. I made no scratch probe of them.
- **-X builds:** print `RuntimeWarning` at moves.py:966 (5 times), as on 75 cm. This is pre-existing.
- **Not done:** ruling 40's sets.py freeze text (no sets.py change was in this job), a merge of the working branch's newer commits, `moves legacy` and `regress check`.

Scratch logs and readouts are in /tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/rebuild60/:
- build_*.log, governor.log, tests_final.log, verify*.log
- readout_*.json, knee_*.json*, probe.log
- the scripts
The pytest temp directories and the probe arrays are deleted.
