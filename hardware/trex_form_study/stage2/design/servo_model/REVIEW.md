<!-- Working paper, 16 Nov 2026: the independent reviewer's report on SERVO_MODEL_DESIGN.md as first written. Nothing here is a rule. Its fixes are not yet in the design note. -->

**Verdict: usable after the listed fixes.** The run-time layer and its three hooks hold up against the code. Section 9's owner-facing text and condition 2's corner rule should not go to the owner as written.

Marks: [P] = printed by my probes today; [calc] = my arithmetic; (summary) = web-search summary only, page not opened. Paths are relative to `hardware/trex_form_study/`; RES, PLANT, LAT, BENCH, DL as in the note.

## Findings, most serious first

### Blockers (for the parts named)

**B1. Section 9 states two things the note's own tables contradict.**
- **Sentence:** "Where the maker gives a value, that value is the nominal."
  - Evidence: `armature` X4 and X6 are nominal at A7's 0.00324 and 0.0043, although the maker gives 0.000082 kg·m² (RES:85) and 0.66 kg·cm² (RES:154).
  - `gain_cap` is nominal "none" against the sibling manual's 500 and 5 (LAT:442). `enc_offset` is nominal 0 against the maker's 0.4° accuracy (RES:85).
  - `fric_c` X4 and X6 take the maker's backdrive torque, a different quantity. `play_deg` takes the maker's upper limit.
- **Sentences:** "the bench (gate 2) replaces them" (section 9) and "unsourced; the bench measures it" (section 3).
  - Evidence: the note's own check row finds no sheet measures gear play (BENCH:327-331 confirmed). N1 to N6 are "none on the sheet today". P32 (DL:47) says the bench is not expected before a build works in simulation.
- **Replace with:** "Dry friction, gear play and the torque-speed line start from the maker's catalogue figures; the rotor inertia, the drives' gain limits and the encoder zero error do not, and the table says why. Values without a source are labelled. The bench stage would replace them only once measurements N1 to N6 are added to its sheet; today the sheet measures delay, stiffness, friction and the tail's heat."

**B2. Condition 2's corners can fail or pass for reasons that are not the robot's, and P40 stops the purchase on a fail (`BUY_GATE_DRAFT.md:50`).**
- **`J+`** rests on an unresolved reading of one catalogue number. The note's own arithmetic puts the knee over its peak there at the animal pace (22.2 against 21.0 N·m).
- **`V-`** (21.0 V with 0.15 Ω) is outside the drives' supply window.
  - Section 2c says the walk's 8.9 A allows 0.11 Ω. With 0.15 Ω the bus sits at 19.3 to 19.7 V [calc], below the 20 V minimum (RES:69).
  - A "stays up" there is on a bus the real drive may cut out on.
  - The pair is not one supply: the tether is 24 V with 0.13 Ω (`round2/refine/MATERIALS.md:233`); no pack resistance has a source.
- **`P+`** was lowered from the draft's 1° to 0.5° ("Top [E]: twice that").
  - The only third-party figure is 4.0 times the maker's: 0.334° against 0.0833° on an RMD X8-V2, a unit the authors call worn (arXiv 2605.24127, summary). That gives 1.0° on a 15-arcmin part.
  - The ankle push rods' play has no source and is not included.
- **`F+`** is "above my bound on purpose".
- **The interior is never judged.**
  - A17 (DL:123) records a non-monotone delay response on this body: it falls at 10 ms and stays up at 20 ms.
  - Judged panels never leave nominal for `enc_offset`, `sense_steps`, `fric_joint`, `kp_scale`, `kv_scale`, or the upper `cmd_lag_ms`.
- **Replace with:**
  - Judge `N` plus a seeded drawn panel at 4 and 10 ms under the bar's own pass rule. Section 4 already defines its seed, `SeedSequence([panel seed, 2])`.
  - Judge only corners whose value has a stated basis: `F-`, `F+` (labelled as above the bound), `P0`, `P+` at 1.0° on the leg drives, `M-`, `M+`.
  - Report `J+`, `V-`, `G5`, `T+` and the X6-50-L setting of M2 until the maker answers.
  - Record a `V-` run with any step below 20 V as "left the supply window", not as stayed up.

### Major

**M1. The dead zone assumes a motor-side loop; for the neck and tail that is probably wrong.**
- Cells: section 2b "What it assumes"; section 3 `play_deg` neck and tail; `P+` "1.5° on the tail".
- Evidence: the cloned small-biped code says of its DYNAMIXEL servo, "the magnetic encoder sits on the OUTPUT side of that play, so the firmware position loop closes on main+backlash" (`/home/user/pollen-robotics/microduck_rl/src/mjlab_microduck/actuator/friction_dr_bam.py:72-73`; `tasks/backlash.py:6-9`).
- I found no source for the sensor's side on the XM540 or the STS3215. The maker's page lists an AS5045 contactless encoder for the XM540 (summary); the side is not stated.
- At `P+` the tail's dead zone is 0.64 N·m of a 0.862 N·m peak [calc].
- Replace: no dead zone and no reading shift on the neck and tail until N6. Say in sections 7 and 9 that the leg drives' dead zone assumes a motor-side loop, which has no source (RES:76 says only "dual encoder").

**M2. The X6's 24 V line is less supported than the row says, and the alternative part is not modelled.**
- Cells: section 1.2 "Torque against speed"; section 3 `ts_corner` X6 "3.36 to 8.01"; section 2c "The catalogue supports about 10 N·m there".
- The note drops reader 3's consistency check (`stage2/design/servo_model/reader3_outside_sources.md` §1.2): 16 Vdc/krpm (RES:154) at 176 rpm is 55 V of back-EMF on a 48 V bus.
- By that figure the no-load speed at 24 V is 8.01 rad/s, which is today's corner. With the note's own copper slope (7.17 N·m·s/rad) [calc]:
  - the peak is available only below 2.15 rad/s;
  - at 5.8 rad/s the drive has 15.8 N·m at 24 V and 11.5 N·m at 22.2 V, against the reference's 16.6 N·m.
- If the X6-50-L ships (RES:154), the maker's peak is 45 N·m (RES:167), so the planning peak is 31.5 N·m, not 42. No draw or corner covers that.
- Replace: low end 2.15 [E from M], or mark it open. Add a reported setting with hip-pitch peak 31.5 N·m and friction 0.2 N·m. Reword to "my line through the two 48 V points, shifted to 24 V, gives about 10 N·m; the catalogue has no 24 V data."

**M3. The tail and neck nominals take from each third-party fit only the part that lowers the load.**
- Cells: section 3 `fric_c` and `fric_v` tail (0.05, 0.06) against `armature` tail "same"; section 8 choice 16.
- The same fit file gives armature 0.0284 (`dl_bam/br_bam_params_feetech_sts3215_7_4V_m1.json`); the small biped's model gives 0.027. Both are for the 7.4 V version.
- I reran the designer's `probe_ref_accel.py` [P]. With 0.0284 the tail's RMS torque on the stored references rises against the 0.491 N·m limit (prescribed motion):

| Reference | Today's armature | With 0.0284 |
|---|---|---|
| nominal walk | 0.20 | 0.25 |
| brisk walk | 0.26 | 0.36 |
| animal pace | 0.30 | 0.41 |

- Replace: read condition 4 at the worst of `N`, `F+` and tail armature 0.028; or make 0.028 the tail's nominal [T, other voltage version].

**M4. The gain copies are specified for `gain_cap` only.**
- Cells: section 2f "kv = 2 √(…)"; section 2d `kp_scale`, `kv_scale`.
- `Plant.command` folds the rate target with the plant's own `kp` and `kv` (`designlab/stage1/plant.py:575-576`, `662-681`). Changing the model's kv alone leaves a torque of (kv_plant − kv_model) × speed.
- At `J+` the knee's kv goes from 11.51 to 19.47; at the nominal walk's 4.2 rad/s that is 33 N·m against a 21 N·m peak [calc].
- The tracker's ankle cap copies both gains when built (`designlab/stage1/control.py:1240-1248`).
- Replace: say per part whether the plant's and the tracker's copies follow (`J+` and `G5` yes; box-servo scales no), and add a test.

**M5. "`bar.py` is not edited" does not survive work packages S3 and S4.**
- Cells: section 5.3 row for `bar.py`; section 6 S3 "the corner panel as a bar task" and S4 "R0 and R2 rows … development episodes"; test T1.
- Tasks are 3 and 4 only (`designlab/stage2/bar.py:140`, `3468`).
- On the three 75 cm sets a non-R0 row runs only as a task's unit; on any other set only R0 rows run (`bar.py:333-340`, `2196-2222`). A build check is one R0 episode on V0.
- Branches keyed on the controller's name would misread served rows:
  - `NO_VERDICT_CONTROLLERS = ("R0",)` (`bar.py:138`): an `R0s…` row would get a verdict.
  - `ATTRIBUTED_CONTROLLERS = ("R2",)` (`bar.py:1603`) and `panels.VERDICT_ROWS` (`designlab/stage1/panels.py:82`).
  - `control.relaxed` (`control.py:2061-2099`) cannot lift play, friction or sag, so fall labels would mislead.
- A new task edits `bar.py` and moves `45719d03`; T1 then fails by design.
- Replace: state which route gate 1's records take (an existing task's budget, a new task with a planned hash change, or a runner outside `bar.py`) and how served rows are named so these branches read them correctly.

**M6. Gear friction is counted against output-side ratings, and A2's 0.70 is spent twice.**
- Cells: section 5.2 trap 4; section 2g "A torque-scale draw".
- The maker's peak, rated torque and torque constant are output values (RES:85, 154: "Kt side: output"). Friction sits outside `actuator_force`.
- At `F+` the net driving torque at the limit is 35.4 N·m on the X6 and 17.3 N·m on the X4 [calc].
- Section 2g uses the same 0.70 to cover a third-party torque constant 33 % low (1.393 against 2.09 N·m/A; summary).
- Replace: say this in sections 2a and 7. Add a torque-scale draw (the envelope already takes `torque_scale`, `designlab/stage1/drive.py:196`), or state that A2's margin is used up.

**M7. The delay table adds up to the budget, not to the judged 10 ms, and the note never relates the two.**
- Cells: section 2d, both tables.
- The sums are right: 0.31 to 2.88 ms, and 3.50 ms with S.
- Under `N` the judged "10 ms" becomes 10 ms plus a lag of 0.5 or 1.2 ms, plus a hold of up to 8 ms on the neck and tail. P40 approved pure delays.
- LAT's D estimate already contains the winding rise ("plus the current's rise", LAT term-basis row), so the lag is counted twice against the budget.
- The bench's reading folds S beyond 2 ms into T (BENCH:93-96), while the note models it as `sense_steps`: twice again when gate 2's value goes in.
- Replace: one sentence on each, and a rule for which part of a measured D and S goes where.

**M8. Five open numbers can be closed by asking the maker; only one is made a precondition.**
- Cells: section 8 choices 1 and 2; section 7 "Parts".
- The five: the side of the gear the inertia figure is on; the motion-mode gain ranges and units; the X6's curve at 24 V; which X6 ships; which encoder closes the loop and which is reported.
- If the gain caps are real, `N` uses gains the drive cannot hold (damping ratio 0.22 to 0.72; I recomputed these).
- Replace: all five before gate 1 is judged.
- I could not read the protocol PDF: `cdn.robotshop.com` is outside this session's allowed hosts.

### Minor

1. **Section 2b, "A joint with less torque asked than that floats."** The integrator is implicitfast [P]. On a one-joint toy with the servo torque exactly zero, the joint gains 29 % less speed per step at kv 11.51 and 41 % less at 19.45, matching h T / (M + h kv) [P]. Add to "What it cannot show": inside the gap the joint carries a virtual inertia of 2 ms × kv.
2. **Section 2c, sag.** The one-step iteration oscillates when R P / V² exceeds 1, and has no solution above V0² / 4R: 735 W at `V-`, 2,464 W at `N` [calc]. Ten X4 at 21 N·m draw 888 W. Use the closed form and count a collapse as a fault, not a clip to 12 V.
3. **Sections 1.3 and 3, "MX-106 dry 0.12".** The file's Coulomb term is 1.0e-5 N·m; 0.122 is its low-speed (Stribeck) term (`dl_bam/pin_bam_params_mx106_m6.json`).
4. **Check table, "identical to the worktree's".** No longer true. The worktree has uncommitted edits (148 changed lines in `r5env.py`); the reset's lines 431 and 504-505 are 468 and 541-542 there. The edits add `RESUME_KEYS` and `config_from(describe())`, which the `servo` field must round-trip through.
5. **Section 5.3, `describe()`.** `R5Config.describe()` is a plain `asdict` (`designlab/stage2/r5env.py:159-160`). `FACTOR_RULE`, `RESET_ORDER` and `DRAW_ORDER` are printed by the env's `describe()` and would be false with the layer on. The plan does not say how they change.
6. **Section 5.3, "In a record key? no" for `r5env.py`.** G8a's result hashes `r5env.py` (`designlab/stage2/g8a.py:254-255`). No record holds it today, but one made before the edit goes stale.
7. **Section 1.1 inertia.** The V4 sibling's 0.3 kg·cm² (RES:52) is unused; it gives 0.039 kg·m² reflected [calc]. The nominal sits at the bottom of a 33-fold range.
8. **`M-`, `M+`.** They are absent from T13, and the note does not say when the mass is applied on the judging side. The layer "acts from the first physics step"; the env applies mass before the start.
9. **Section 4, seed.** With `options["seeds"]` and no `episode_seed` the episode seed is None (`r5env.py:478-480`). Say a named setting is then required.
10. **Ankle.** The play and the offset are on motor coordinates; the encoders are joint sensors [P]. State the mapping.
11. **Section 2c, "today's rule … applied every step".** Today's rule also scales the corner (`designlab/catalog.py:77-82`); the note's shifts the line. They are equal at 24.0 V only.
12. **Choice 9.** With condition 5 it is 640 runs, so 53 % at a 99.9 % true rate [calc], not 73 %.
13. **Names.** `R2s` is already a registered controller (`designlab/stage1/speedfit.py:1213`). Section 9 cites `DESIGN.md`; the file is `SERVO_MODEL_DESIGN.md`.
14. **Scope.** `stage2/jetson75/BUDGET_75.md:312-314` says the build's head "as decided today has the neck-yaw servo only". That servo has no row. The 20.3 W auxiliary figure is the Pi 5 record's, not the Jetson's.

### The owner's question

The note does not say it plainly. Section 7 opens well, but nothing states what a pass means. Proposed text:

> "A pass tells you a policy trained twice keeps the simulated body walking at 0.30 m/s with friction, play, a lower torque-speed line, a sagging bus and delay anywhere in the table's ranges. It does not tell you the real parts are inside those ranges. Of the table's 38 rows, 7 nominals are the maker's figure for the part, 5 come from a third party's fit of another model or voltage version, and 26 are the study's assumptions or estimates [my count]. None is measured on this build."

## Checked and found right

- **Maker's figures:** every one at its RES line (63-76, 85, 131-144, 154, 186, 1491-1502, 1554-1567), and the PLANT values. Actuator names and counts match `BUDGET_75.md:380` and BENCH:322-398 (10 X4, 2 X6, one XM540, one STS3215).
- **Arithmetic:** the torque-speed lines (5.76, 4.16, 3.36; 7.4 and 0.4 N·m in T7), the efficiency bounds (3.16, 4.05), the friction sums, c_cu (0.185, 0.139), L/R, the damping ratios, kp × play, the inertia ratios, and 0.11 Ω.
- **Third-party files:** the STS3215 fit, the firmware law (kp 32, 0.166, 3400 steps/s) and the small biped's values.
- **Hooks:**
  - `Plant.step` reads the envelope before each step.
  - `servo_delay` refuses only a replaced `command` or `reset`.
  - The row builds its tracker inside the episode; the env builds its trackers at construction.
  - The judge reads encoders only through the estimator.
  - The actor's observation comes from the estimator, so a patched reading reaches the policy.
  - The nesting order (delay, then layer) is the same on both sides.
- **Hashes:** `c0027992` and `45719d03` [P]. No hashed file is in the edit list. `designlab/stage2/` is globbed by no hash.
- **Model:** `tendon_armature` exists and is 0.00324 on the four ankle tendons; 14 friction rows; solref 0.02 [P].
- **Units:** consistent throughout.

## Not checked

- Any maker page.
- The arXiv paper itself (summary only), including its 10 Hz torque bandwidth.
- The designer's timing probe and three of its other probes (not rerun).
- Reader 2's report in full; LAT beyond the baseline wiring; the 60 cm body.
- Whether tendon friction behaves as claimed at run time.
- No episode was run.

Deviations from the standing rules: one physics step on a one-joint toy model (not the robot), and one arithmetic one-liner run with the system python outside `tools/lowprio.sh`.

My scripts are in `/tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/servo_design/reviewer/`:
- probe_model.py
- probe_hashes.py
- probe_toy_deadzone.py
