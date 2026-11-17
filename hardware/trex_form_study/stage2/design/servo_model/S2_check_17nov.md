**Verdict: merge as it stands.** I found no defect that blocks the merge. The items below are minor: one design sentence is the lead's to rule on, one test could be tighter (optional), and the rest were already flagged for the lead or for S3/S4. Simulation only: the servo values are a model, and nothing here was measured. I made no records.

**Merge readiness (read this first)**
- The working branch moved while I was reviewing. It went from 4a9d2aad to **23ea3fc1** ("Ruling 40 on LT3"; local branch and origin agree). The new commits change r5lib.py, gates2.py, barviews.py, their tests and some docs. They touch none of S2's files.
- So the branch no longer fast-forwards; merging it needs a merge commit.
- `git merge-tree --write-tree claude/mesozoic-trex-stage2-refine-4jffre HEAD` exits 0 with tree ffec0291e776f49036f06d559584b94c7dc353ca and no conflicts.
- r5lib.py is the only changed module that the r5env and g8s tests import. With the tip's r5lib.py laid over the S2 code, those tests gave `71 passed, 3 skipped in 95.58s (0:01:35)`.
- The bar's and panels' hashed files (bar.py, readings.py, setio.py, speedfit.py, stage1/*) are not in the tip's diff, so the hashes should stay the registered pair after the merge. I did not print them on the merged tree.
- **Tip to merge: f94f61cb992bc1b62c248ac0f4de8cde28e9395b.** All 4 commits (cc024ffd, b93d8b44, cba352ac, f94f61cb) end with exactly the two attribution lines. No commit names an AI model elsewhere; the only matches for "model" are "servo model".

**Findings, most serious first**
1. **Design text left over (for the lead).** Section 5.1, the "Training env" row, still has the first wording: "...so the run's gains and their copies must already be set (2d; ...)". With the split, only the copies are set at entry; the model gains are set at the first physics step. This sentence is outside the three amendments, so the builder rightly left it. The lead may want it amended to match 2d.
   - The design's `r5env.py:NNN` line references point to the file before S2. This is cosmetic.
2. **Small test gap (optional fix).** The env's early refusal of a "draw" when `options.seeds` is given without an `episode_seed` is not pinned by any test.
   - My mutant that removes it (e11) passes every test. `servo.draw(None)` still raises later, at step 8, and its message also matches `match="episode seed"`.
   - What changes is when the refusal comes: after the new episode's factors and arrays are applied, with the previous episode's delay still in.
   - Fix: in `test_the_option_draws_nothing_from_the_envs_stream`, match `SERVO_RULE`, which only the env's message contains.
3. **Mid-start forward pass (the builder's item (c); the lead's call).** I confirmed it.
   - The model arrays in that one `mj_forward` are today's, but the ctrl in it was folded with the run's copies. The force therefore carries an extra (kv_run − kv_today)·q̇* term, and it feeds the estimator's first reading.
   - Every training draw moves the kv copies. In my probe, train/0, 1, 7 and 12345 move the ankle motors' kv from 2.08 to between 2.19 and 2.86. So this applies to every mid start under `servo="draw"`. It happens once per episode, and no command is served in that pass.
   - SERVO_RULE's "forward pass ... today's model's" is true of the model arrays only.
4. **Delay-before-layer order cannot be seen in the digests.** My mutant e16 enters the layer before the delay and passes every test. That is expected (trap 2: either order works), so this is a note, not a defect.
5. **T14's settings.** The T14 env test uses "today" plus J+'s inertia, "today" plus the gain cap, and "today" plus N's inertia, not the full J+, G5 and N settings. This is the same form as S1's plant-level T14. With the full settings the lag, play, offset and hold change the command, so the formula would not apply as written. I think this reading is defensible; it is listed.
6. **For S3/S4, as the builder said.**
   - s2rows.py:623-637 builds its env from the run's `env_configs[0]` and resets with `options.seeds` but no episode seed. A run trained with `servo="draw"` would be refused there.
   - G8s compares against its own `ServedRow`; it must be run again against S3's `servo.Served`.
   - Merging moves `train2.sources()`, so runs started before S2 resume only with `--allow-source-change`.
   - Design trap 13 says G8a is run again after S2. That has not been done (I ran only G8a's tests). My grep found no stored G8a result, only one review JSON that names g8a.

**What I checked and found right**
- **Hashes:** `45719d030359ec7d5d3914af3814d837fdc05d77652ba45c64792b511027528d c0027992de21067525cc4489b74acb157feb89f3a4c7618580addc2b341be1c1` (the registered pair). TABLE_SHA256 is 342f3948cef116ea7fde49106817596a3d69ffa8d85c84dcb7c35f442546ef57; servo.py is not in the diff.
- **Only the package's 7 files differ** from the working branch: g8s.py, r5env.py, train2.py, SERVO_MODEL_DESIGN.md, and tests g8s, r5env and train2. servo.py, bar.py and moves.py are untouched. The worktree and the main checkout are both clean (`git status --porcelain` empty). The only ignored files are the restored inputs.
- **Default off, bit for bit (my own probe).** In one process I loaded r5env.py from 4a9d2aad beside HEAD's and compared four S1 cases, 4,270 physics steps in all:
  - quiet R5, a mid start with k5 and a stance start with k3: 600 steps
  - default R5, seed 7, 4 episodes: 1,600 steps
  - default R5 with mid_start 1, 3 episodes: 1,200 steps
  - default R6, seed 3, 3 episodes: 870 steps
  - The state-digest chain, observations, rewards, terminated/truncated, info, `env.describe()` and the config's `describe()` are identical in every case.
- **Option on:**
  - The layer is left first in reset and entered after the delay, before the mid-start queue fill.
  - The trunk mass of M-, M+, W1 and W2 becomes the episode's mass factor.
  - joint_damping and joint_friction must be 1, and the env's draws stay, so its stream does not shift.
  - The training draw is `SeedSequence([ep, 2])`; "D/<s>" gives `panel_draw(s)`.
  - `describe()` drops `servo` when it is None, and gives the servo variants of the three rule texts with the table's sha256 when it is on.
  - These all match design 5.1, 5.2 (traps 9 and 12), 5.3, section 4 and T12's env part.
- **The 2d amendment and the note to ruling 21 match the code.**
  - `control.py:1240` copies `p.kp`/`p.kv` into the ankle cap when the tracker is built. `panels.py:844` builds the tracker inside the episode, and `panels.py:880-881` gives R0 no cap.
  - My probe of every named setting: only J+ and W1 move the ankle motors' kv copy (11.284 against 2.081/2.083). G5 does not.
  - No judged setting moves an ankle gain: F-, F+, P0, P+, M-, M+, and D at seeds 3042 to 3061.
- **Section 3:** in servo.py the five cells are all "the study's assumption", and the design's wording now matches the code's source texts.
- **Trap 8** matches `Layer.enter` and `Layer.exit`.
- **g8s** has G8a's form with the layer on both sides, the trunk mass applied before the start on both sides, and T13's settings at k = 0 and 5.
- **Mutants:** I made my own scratch copies of the changed files only and overlaid them on the package path. The unchanged copy passes (9 selected r5env tests, 27 g8s, test_sources).

| Mutant | r5env servo tests | g8s tests |
|---|---|---|
| e01 layer entered after the mid fill | 3 failed (T14) | 27 passed |
| e03 layer left late | 5 failed | 25 failed |
| e04 no leave at close | 1 failed | 20 failed |
| e05 describe keeps `servo: None` | 2 failed | 27 passed |
| e06 setting's mass not the episode's | 1 failed | 27 passed |
| e07 no joint check at reset | 1 failed | 27 passed |
| e08 no joint check on the config | 1 failed | 27 passed |
| e09 draw shifts the env's stream | 1 failed | 27 passed |
| e10 wrong draw tag | 2 failed | 27 passed |
| e11 no early refusal | **all pass** (finding 2) | 27 passed |
| e12 D/<s> uses the wrong seed | 1 failed | 27 passed |
| e13 old rule texts printed with the option on | 1 failed | 27 passed |
| e14 options.servo ignored | 5 failed | 22 failed |
| e15 options.servo allowed with the option off | 1 failed | 27 passed |
| e16 layer before the delay | all pass (equivalent) | 27 passed |

  - g8s mutants, g8s tests: g01 no trunk mass on the row, 6 failed; g02 row not served, 22 failed; g03 env always at N, 18 failed; g04 no delay on the row side, 11 failed.
  - t01, SOURCE_FILES without servo and g8s: test_sources `1 failed`.
  - Option-off mutants against the bit-for-bit test: e17 (an extra draw from the env's stream with the option off) `1 failed`; e18 (the "today" layer always in) `1 failed`.

**Exact test summary lines (my runs, in the worktree)**
- `tests/test_stage2_servo.py`: `58 passed in 2.50s`
- `tests/test_stage2_g8s.py`: `27 passed in 19.85s`
- `tests/test_stage2_r5env.py`: `44 passed, 3 skipped in 82.64s (0:01:22)`. The skips: two need the group-x inputs, one is the slow test.
- `tests/test_stage2_train2.py`, the 8 tests that read the sources or the allowance (test_sources, both resume-refusal tests, the allowance, halt, monitor, pass-through and allowance-from-record tests): `8 passed, 52 deselected in 67.03s (0:01:07)`
- `tests/test_stage2_g8a.py`: `8 passed in 12.51s`
- r5env and g8s tests with the tip's r5lib.py overlaid: `71 passed, 3 skipped in 95.58s (0:01:35)`
- ruff on the 6 changed .py files: `All checks passed!`

**What I did not check**
- The full test suite, and the other 52 train2 tests.
- The vecpool, s2rows and nb tests.
- A full-length G8s or `bar_form` on any real set (no runs that count).
- G8s on X6L, G5, T+, W1 or W2.
- The hashes printed on the merged tree itself; I reasoned from its file list.
- An exact replay of the builder's own M1 to M6 mutants (I made my own).
- S3 and S4.

My scripts and outputs are in /tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/servo_s2/check/: probe_off.py and probe_off.json, probe_gains.py, make_mutants.py, run_mut.sh, overlay_plugin.py and mut_results.txt. I deleted the pytest temp directories, the mutant copies and the exported base files.
