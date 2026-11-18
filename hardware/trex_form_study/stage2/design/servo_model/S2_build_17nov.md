S2 report (servo design package S2). Worktree /home/user/mesozoic-wt/wpc-s1, branch wt/wpc-s1. Simulation only: the servo values are a model, nothing was measured, and no episode here is a record.

**1. Merge of the working branch**
- The worktree head 3f239d42 was an ancestor of claude/mesozoic-trex-stage2-refine-4jffre (4a9d2aad), so `git merge --ff-only` fast-forwarded to 4a9d2aad. There was no merge commit and no conflict.
- The working branch was still at 4a9d2aad at the end, and is an ancestor of HEAD.
- Inputs check: `picks-models 6 6 0 0 agree`, `picks-arrays 84 84 0 0 agree`.
- servo.py is unchanged; S2 needed no new hook from it.

**2. Per point**

(1) The env option, in /home/user/mesozoic-wt/wpc-s1/hardware/trex_form_study/designlab/stage2/r5env.py
- `R5Config.servo` defaults to None (off). The values that turn it on are "draw", a name in servo.NAMED other than D, or "D/<bar seed>". `options["servo"]` can also be a `servo.ServoParams`.
- `describe()` leaves the key out when it is None. `config_from` reads it back.
- Reset order:
  - The layer is left first thing in reset, before the arrays are restored (trap 9).
  - The episode's parameters are fixed after the episode seed. A draw uses `SeedSequence([ep, 2])` and is refused when `options.seeds` is given without an `episode_seed`.
  - The layer is entered with `servo.served` after the delay is entered and before a mid-task start fills the queue.
- Factors:
  - joint_damping and joint_friction must be 1. With randomize on, their ranges must be (1, 1). The draws stay, so the stream does not shift.
  - A named setting's trunk mass factor becomes the episode's mass factor, applied before the start.
- Rule texts: `SERVO_RULE`, `SERVO_RESET_ORDER`, `SERVO_DRAW_ORDER` (it carries TABLE_SHA256) and `SERVO_FACTOR_RULE` are new; the old strings are not edited. `describe()` prints the servo variants and adds a `servo` block only when the option is on.
- Tests that cover it: `test_the_option_on_runs_the_layer_after_the_delay_and_leaves_it_first`, `test_the_options_refusals_before_anything_is_built`, `test_the_option_draws_nothing_from_the_envs_stream` (T12's env part).
- Default-off test: `test_the_option_off_is_todays_env_bit_for_bit`. It pins two episodes' state-digest hashes and two config hashes, computed on r5env.py at 4a9d2aad.

(2) /home/user/mesozoic-wt/wpc-s1/hardware/trex_form_study/designlab/stage2/g8s.py (new)
- It is G8a's form with the servo layer on both sides and one parameter object (`ALIGN_RULE`).
- `stock` and `check` run on the S1 view. `bar_form` runs on a frozen set through `bar.Delayed`.
- The judging side is `ServedRow`: the setting's trunk mass is applied before the inner episode and undone after, and the inner episode runs inside `servo.served`.

(3) Tests, in /home/user/mesozoic-wt/wpc-s1/hardware/trex_form_study/tests/test_stage2_g8s.py and test_stage2_r5env.py
- T1: both hashes are pinned, none of their files is servo.py, g8s.py or r5env.py, and `regress.check() == []`.
- T2: covered in test_stage2_r5env.py, and G8a's own tests pass unchanged.
- T3: `today` on both sides at 0 and 6 ms equals the run without the layer.
- T13: N, F+, P+, V-, J+, M-, M+ and D at seeds 3042, 3043 and 3044, at k = 0 and 5. All 20 are identical over 500 physics steps.
- T14's mid-start delay-queue case, under J+, G5 and the tail at N:
  - The queued commands are folded with the run's gain copies.
  - At each of the first k physics steps the force equals kp(q* − q) + kv(q̇* − q̇) + τ with the run's gains.
- Also added: a sensitivity test (another setting differs at the first step), a test that the plant is given back, and a bar_form test on a synthetic T0 set.

How the tests fail without each change. Each mutant undid one change; the file was restored afterwards and checked byte for byte with sha256.
- M1, layer entered after the queue fill: `3 failed in 8.31s` (the T14 fold assertion).
- M2, layer left at step 10 instead of first: `1 failed in 3.17s` (mj_setConst fields differ).
- M3, the setting's mass not made the episode's: `1 failed, 27 passed in 25.47s`. G8s passes because its env episode names the mass itself.
- M4, describe keeps `servo: None`: `2 failed in 1.11s`.
- M5, the joint-factor check removed: `1 failed in 5.79s`.
- M6, SOURCE_FILES without servo and g8s: `1 failed in 1.62s`.

(4) `train2.SOURCE_FILES` gains "servo" and "g8s". `tests/test_stage2_train2.py::test_sources` now names both. This moves `train2.sources()`, as any edit to r5env.py would: a run started before S2 resumes only with --allow-source-change.

(5) Design amendments, /home/user/mesozoic-wt/wpc-s1/hardware/trex_form_study/stage2/design/servo_model/SERVO_MODEL_DESIGN.md
- Section 2d's gain-copies table, armature row, tracker column now says what the code does:
  - The copies are set at the layer's entry and the model gains at the first physics step.
  - A served R2 row whose tracker is built inside the episode takes the run's kv in its ankle cap at J+ and W1 only (11.28 against 2.08, which I printed again from `servo.Layer`). Both settings are reported only.
- The rule sentence under that table said "in the same call". I amended it too, because it contradicts the split that was kept. This is my reading of "the gain-copies table".
- The note to ruling 21 (section 8) is amended to match.
- Section 3: the five cells now name their kind as the study's assumption [A], and one sentence above the table records this. The counts do not change, and TABLE_SHA256 is still 342f3948cef116ea7fde49106817596a3d69ffa8d85c84dcb7c35f442546ef57.
- Trap 8: the layer refuses to enter over a step hook it did not install.

**For the lead** (each is built as the first option)
- a) Option values:
  - "draw" for the per-episode training draw; "train" was the alternative name.
  - Configs take strings only; `options` also take a `ServoParams`, which is the "one parameter object".
- b) joint_damping/joint_friction: the env refuses configs and options that are not (1, 1). The alternative is to force them to 1 silently.
- c) Mid-task start forward pass, under a setting that moves a gain copy (the tail at N, J+, W1, G5, every training draw):
  - The first reading's forward pass runs on today's model arrays, with the oldest queued command folded with the run's copies. For that one pass it mixes kv values on those drives; no command is served in it.
  - Alternatives are to fold that pass's command with today's copies, or to write the model gains at entry (which breaks G8s's reset, per S1). Recommend as built.
- d) G8s compares against its own `ServedRow`, because S3's `servo.Served` is not built. G8s should be run again against S3's row. The named settings use seed 3042 and D uses seeds 3042–3044; the design names no seeds.
- e) The reset info gains `servo` (the setting's name) only when the option is on.
- f) For S3/S4: `s2rows`' equivalence check resets with `options.seeds` and no episode seed. A run whose config has servo="draw" would be refused there.

**3. Outputs**
- `tests/test_stage2_servo.py`: `58 passed in 2.83s`
- `tests/test_stage2_g8s.py`: `27 passed in 21.25s`
- `tests/test_stage2_r5env.py`: `44 passed, 3 skipped in 84.59s (0:01:24)`. The skips were already there: two need the group-x inputs and one is the slow test.
- train2 tests that read the sources (test_sources, the resume-refusal test, the allowance test, the halt test, the resume-order test): `5 passed, 55 deselected in 28.10s`
- `tests/test_stage2_g8a.py`: `8 passed in 12.21s`
- ruff: `All checks passed!`
- Hashes: `45719d030359ec7d5d3914af3814d837fdc05d77652ba45c64792b511027528d c0027992de21067525cc4489b74acb157feb89f3a4c7618580addc2b341be1c1`, the registered pair.
- Default-off evidence:
  - My probe ran before the change and at HEAD f94f61cb: 3 quiet episodes plus 4 default-config episodes, hashing observations, rewards, reset info and every state digest, plus the description hashes. The two outputs are byte-identical, and the pre-change run was repeatable.
  - In one process, the base r5env.py taken from git at 4a9d2aad and HEAD give equal state-digest hashes (371c9020…, d208abed…) and equal config descriptions.
- Not done: the full test suite, an independent review, G8s on X6L/G5/T+/W1/W2, anything on a look set, and S3/S4.

**4. Diff and status**
`git diff claude/mesozoic-trex-stage2-refine-4jffre --stat`:
```
 hardware/trex_form_study/designlab/stage2/g8s.py   | 278 +++++++++++++++++++++
 hardware/trex_form_study/designlab/stage2/r5env.py | 198 ++++++++++++++-
 .../trex_form_study/designlab/stage2/train2.py     |   5 +-
 .../design/servo_model/SERVO_MODEL_DESIGN.md       |  20 +-
 hardware/trex_form_study/tests/test_stage2_g8s.py  | 217 ++++++++++++++++
 .../trex_form_study/tests/test_stage2_r5env.py     | 246 +++++++++++++++++-
 .../trex_form_study/tests/test_stage2_train2.py    |   2 +-
 7 files changed, 938 insertions(+), 28 deletions(-)
```
- `git status --porcelain`: empty. No caches are left in the worktree, and the main checkout is clean.
- I deleted the pytest temp directories and the scratch export. The probe scripts and outputs (small) are kept in /tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/servo_s2/.

**5. Commits, oldest first**
- cc024ffd Servo model S2: the env's servo option (R5Config.servo, off by default)
- b93d8b44 Servo model S2: gate G8s, the env with the servo option against the served R0 row
- cba352ac Servo model S2: train2.SOURCE_FILES gains servo and g8s (third list, point 8)
- f94f61cb Servo model design: the three amendments of the lead's third list

Branch head: f94f61cb992bc1b62c248ac0f4de8cde28e9395b. Nothing was pushed.
