# build:S1 (17 Nov): the agent's final report

Working paper. Nothing here is a rule.

Package S1 (servo model, gate 1) final report. Worktree /home/user/mesozoic-wt/wpc-s1, branch wt/wpc-s1. This is simulation only: no value in the code was measured on hardware, and nothing it ran is a result or a record.

**1. Merge of the working branch**
- Nothing to merge. The worktree started at fccebe42, which is the tip of claude/mesozoic-trex-stage2-refine-4jffre. I checked again just before committing: the working branch is an ancestor of HEAD.
- No conflicts and no merge commit.
- Inputs: I put the git-ignored model files and arrays in place with `tools/restore_inputs.py copy-from /home/user/mesozoic-labs --group picks-models --group picks-arrays`. It printed "picks-models 6 6 0 0 agree" and "picks-arrays 84 84 0 0 agree".
- `git check-ignore -v` on those paths shows `round2/.gitignore:3:models/`, `stage1/.gitignore:3:*.npz` and `stage1/.gitignore:9:refs/`.
- The main checkout's `git status --porcelain` is empty.

**2. What I built**

**File `hardware/trex_form_study/designlab/stage2/servo.py` (new, 1310 lines)**
- **`TABLE` (line 112):** the design's 39 rows. Each row has today's value, nominal, range, distribution, source, kind and what would replace it. `KINDS` are the brief's five.
  - `kind_counts()` (266) gives: maker's catalogue value 1, third-party 6, the study's assumption 19, estimate 4, no source 9.
  - 4 rows are flagged as a maker's figure for a related quantity taken as the value. That matches the design's 1 + 4 + 6 + 28.
- **`CONSTANTS` (275):** the layer's fixed numbers, each with source and kind: c_cu from catalog Kt and R, drive efficiency 0.92 (A6), the record's auxiliary input, the 20 V floor, the 10 ms filter, the 5-step hold, the 1.20 rad/s gap, the gain caps, the X6L values.
- **`DRAW_RULE` (311) and `TABLE_SHA256` (590):** the hash covers the table, constants, draw rule and named settings. Final value: 342f3948cef116ea7fde49106817596a3d69ffa8d85c84dcb7c35f442546ef57.
- **`ServoParams` (333):** frozen; validates its fields.
- **`nominal()` (421) and `setting(name)` (501):** all of section 4's named settings (`NAMED_USE` 474). `JUDGED` and `REPORTED` follow P41.
- **`draw(seed, tag)` (514):** draws every parameter every time in a fixed order. The X6's zero-torque speed is drawn after the corner, on [max(8.01, corner + 1.20), 9.215] (the lead's second list). `X6_ZERO_DRAWN = True` (302).
  - `training_draw(e)` (554) uses SeedSequence([e, 2]) and raises on None.
  - `panel_draw(s)` (559) uses SeedSequence([s, 3]), then sets armature and bus_r back to nominal. Latency is not drawn and the trunk mass stays 1.
- **Pure arithmetic:** `bus_voltage` 595 (higher root; V0/2 plus a collapse flag), `line_limits` 612, `friction_loss` 623, `dead_zone` 628, `lag_gain` 638, `kv_rule` 644.
- **The layer:** `served(p, params)` (1234) wraps `Layer` (783), with rules in `LAYER_RULE` 654 and `HOOKS_RULE` 675.
  - At entry it replaces `p.envelope` with `ServoEnvelope` (711), which keeps the nominal peak, rated, no_load and moment for the judge.
  - It sets `p.step` as an instance attribute, and refuses if one is already there.
  - It sets the plant's gain copies `p.kp` and `p.kv`.
  - It writes the model gains, damping, friction and armature (then mj_setConst on a scratch MjData) at the first physics step after entry or after a reset (`_start` 1037). A reset inside restores those arrays and arms again (`_bind` 972).
  - Each step: `_before` (1100) does sag, then the line (`_line` 1073), then friction with the load filter, then the box hold, lag, offset and dead zone, into ctrl. `_after` (1179) puts the controller's command back and patches the readings: error mapping at the ankle, feedback age, the box servos' refresh.
  - Exit (1018) restores every array in `LAYER_FIELDS` and `SETCONST_FIELDS` (693) plus the stat fields, the envelope object, the step hook and the gain copies.
  - `readings()` (926) reports the minimum and mean bus voltage, steps below 20 V, collapse steps, left_supply_window, and each leg drive's share of steps inside its play.
- **Description:** `describe_lines` (1296) and `Layer.describe()` print every row's value, source, kind and replacement, with an UNSOURCED tag on no-source rows.

**File `hardware/trex_form_study/tests/test_stage2_servo.py` (new, 747 lines, 32 functions, 52 cases)**
- **T4:** exact undo for every named setting, D/3042 and train/7, including a reset inside, an error inside, the delay in either order, and the refusals (another layer, an existing step hook, a scaled envelope). Removing the step hook inside is refused and restored. A mass setting is refused while the trunk is as built.
- **Plant-level T3:** `today` is bit-identical to the plant per step, also past the corner. Also checked: the hooks arm at entry and act at the first step.
- **T5:** friction arrays equal the formula; break-away on a one-joint model.
- **T6:** the force equals the dead zone at random states; static deflection on one joint is τ/kp + b; the ankle readings carry (e_a ± e_b)/2; no dead zone on the neck or tail.
- **T7:** the design's 7.4 N·m and 0.4 N·m examples; the sag root and collapse at 735 W and 2,464 W; the supply-window mark.
- **T8:** lag rises as 1 − exp(−n dt/T); the box servos refresh at k = 0 and k ≡ φ (mod 5).
- **T9:** a servo rests at target − o and reads the target; the feedback-age buffer.
- **T10:** the mass-matrix diagonal rises by ΔJ; kv follows the plant's rule (tail 3.59 against today's 2.72); mj_setConst after undo; the setConst fields are complete.
- **T11:** the judge sees nominal limits under every setting.
- **T12:** the draws (same seed, same set; 40 distinct sets over the two blocks; D differs from the training draw; inside ranges; no seed, no draw).
- **T14:** gain copies under J+, G5 and the tail at N; the box servos' scales leave the copies nominal.

**For the lead** (my reading of each is what is built):
- **(a) When the gains are set.** The plant's copies are set at entry; the model's gains at the first physics step. The design's 2d says "in the same call". My split keeps its purpose (no command is served by a kv other than the one it was folded with) and also keeps the reset's forward pass today's model on both sides, which G8s's per-step digests need for D and G5. The alternative is both at entry. Recommend: keep the split.
- **(b) The box servos' hold.** The first command is taken at step 0 whatever the phase, then every 5th step from the phase. The lag acts on the held target. The alternative is to hold the reset's target until the first phased step.
- **(c) Exact "today" values.** The X4 and X6 corner tops and the X6 zero's nominal and top use today's exact values (8.6917, 8.0111, 9.21534) where the design prints 8.69, 8.01 and 9.215. "Nominal = A7" is read as "as found in the model". The MJCF writes armature to 5 significant digits (0.00020092 against the catalog's 0.000200924), so the layer's armature check uses rtol 1e-4.
- **(d) X6L's line.** X6L keeps N's corner and zero with a 31.5 N·m peak, so its slope follows from those. The alternative is a line of its own for that part.
- **(e) Load filter start.** It starts at |τ| from the reset's forward pass, not at 0.

**What S2 must do with this:**
- Enter the layer after the delay and before the mid-start queue fill.
- Pass the `servo` field through the env's config.
- Check that the joint_damping and joint_friction factors are 1. The layer overwrites the drives' damping and friction arrays but does not check those factors.

**3. Outputs I saw**
- Final run of the servo tests: `52 passed in 2.17s`.
- Hash command: `45719d030359ec7d5d3914af3814d837fdc05d77652ba45c64792b511027528d c0027992de21067525cc4489b74acb157feb89f3a4c7618580addc2b341be1c1`, the registered pair.
- ruff on both files: "All checks passed!".
- Acceptance probe (`/tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/wave6/s1/probe_final.txt`): stand/hard, R0 row (zero action), 300 physics steps per setting, seeds of bar seed 3042.
  - "exact undo yes" for today, N, F-, F+, P0, P+, M-, M+ (mass applied by the probe with env.set_trunk_mass and undone), V-, J+, G5, T+, X6L, W1, W2, D/3042, D/3043, D/3044.
  - "the stand row without the layer after all settings: every physics step's digest as before: True".
  - No setting fell. No step went below 20 V; V- and W1 bottomed at 20.723 V.
  - Cost per step, baseline 89.3 µs per physics step without the layer:

| Setting | Layer's own arithmetic (µs/step) | All-in difference (µs/step) |
|---|---|---|
| today | 6.0 | 5.6 |
| N | 39.6 | 35.4 |
| range over all 18 settings | 29.5 to 43.6 | 30.2 to 56.8 |

  - The all-in difference includes the solver's response to the changed model. A tight-loop benchmark under N gave about 30 µs per step, against the design's 28 µs estimate. Load average was about 2 during the runs, so these timings are noisy.
- Not done: T1–T3 and T13; the env part of T12; T14's delay-queue case (all S2); the full test suite (only the new file was run); no independent review.

**4. Diff and status**
`git diff claude/mesozoic-trex-stage2-refine-4jffre --stat`:
```
 hardware/trex_form_study/designlab/stage2/servo.py | 1310 ++++++++++++++++++++
 .../trex_form_study/tests/test_stage2_servo.py     |  747 +++++++++++
 2 files changed, 2057 insertions(+)
```
`git status --porcelain`: empty. Probe and bench scripts and outputs are in the scratchpad under wave6/s1, outside the repository. I removed two git-ignored `__pycache__` directories that my own `python -c` checks had made.

**5. Commits (oldest first)**
- b68648d2 Servo model S1: the parameter table, draw, named settings and run-time layer
- 6afd4f21 Servo model S1: tests T4 to T12 and T14 on the plant

Branch head: 6afd4f215c60fd0976d9842ae717b9747e8d99b6. Nothing pushed.
