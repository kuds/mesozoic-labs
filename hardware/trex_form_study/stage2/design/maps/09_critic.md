# Map 09: completeness critic of maps 01-08

Made 14 Nov 2026 at commit `167dce4` (tree clean before and after). All eight maps were read in full. Paths are
relative to `/home/user/mesozoic-labs/hardware/trex_form_study`. Nothing in the checkout was edited; no pytest.

Labels: **[read]** I read the cited lines myself; **[critic run]** one short kinematic check I ran (section 3; not
a record, do not cite as one); **[arithmetic]** mine; **open** means not settled by any map or by my reading.

Line-number warning: `STAGE2_PLAN.md` gained about 21 lines between `7e9ea0d` and `167dce4`. Maps 02, 03 and 05
cite it 10 to 21 lines low. Current lines: R2L B.2 `:124`, G8a/G9 B.3 `:128`, S2-5 `:134-150`, "every limit" `:180`,
the bar's four rows `:196-199` (walk, turn, nudge, stop), "quick turn needs 1.1-1.4 of range" `:207`, four printed
lines `:231`, G8b `:254`. Map 08's plan lines are current.

## 0. Verdicts

| Item | Enough to start? | Why |
|---|---|---|
| A transitions for any GaitSpec | **partly** | Yes for F25's walks (they are not look specs; every function below `generate` takes a `Gait`). No design for look fields in a transition (V6: F5, F14); maps list what breaks, not how to build it |
| B look-aware pi/4 turns, thigh twist | **partly** | The range check is fully mapped. The finite 90 degree turn is not: two maps disagree on the interface, and the 75 cm turn does not fit the range on any 0.40 m/s walk tried |
| C new bar readings | **yes** for code access; four rules are undefined | Recorder data, hooks and stop events are mapped; the drift line, the turn's command instant, the own-swing "low point" and the stop's count window are rules to pre-register |
| D F25 rungs, ladder and screen | **yes** | Maps 01 section 8 and 07 section 1.7 give the route and the tests that notice. Three decisions remain (names and speed keys, merge or separate directory, how F9 is expressed) |
| E official panels, R2/42 and R2/43, 0-10 ms, two grips | **yes** | Row side and runner options are mapped; the inputs-directory contract is closed below. One rule open (the tracker's omega on a 0.05 H crouch) |
| F R5 env beyond 14 joints, G8a, G9 | **partly** | Widths and G8a are mapped line by line. G9 is only named; the look reference as a tracker reference is not mapped; mid-task starts and command inputs have no design |
| G R2L refit, fixed delay draws | **partly** | The delay draw is mapped; the fit itself (`speedfit.fit`, `Evaluator`, `tune.score`, `speed_g5`) is in no map, and R2L's reference (V6 without F5) does not exist as a rung |
| H notebook, bundle, resume | **yes** | Map 08 is complete on what exists. What remains is undefined text and Colab-side unknowns, not code reading |

## 1. Contradictions between maps

| # | Maps | Contradiction | Resolution |
|---|---|---|---|
| X1 | 03 (intro, 0.3) against 01 (5.5), 06 (3) | 03 says the 75 cm candidate "sets `tail_phase`, a look field, so `turning.generate` refuses it". 01 and 06 say the PR's spec flips the sign of `tail_amp` and is not a look spec | **Closed.** `NEXT_SESSION_PROMPT.md:129-130` fixes the references as "F25 + F9 as the PR's `_tuned_f9_spec` makes it" and `F25_F9_cl` = "that spec with duty 0.59 and lift 0.095 H". Both give `gait.look_active == False` [critic run], so `turning.generate` accepts them and an 11-field `transitions.Gait` carries them whole. 03's statement holds only if someone re-expresses F9 as `tail_phase` |
| X2 | 03 (6, 13) against 02 (2.2, 7) | 03: "`plan()` already takes any phase function for `w`". 02: `plan` takes `Gait` objects | **Closed, 02 is right.** `plan(name, a, b, geo, n)` computes `kind_of(a.name, b.name)`, then `_param_fns(kind, a, b)` and `WINDOWS[kind]` itself (`transitions.py:296-300`; `_param_fns` at `:256`) [read]. There is no argument for phase functions. A rise-and-fall `w(phase)` needs a new plan function, or the module globals `_param_fns` and `WINDOWS` replaced at call time |
| X3 | 01 (8.2) against 07 (1.5, 13) | 01: `screen.run` on a separate directory needs V0, V2, V4, V5. 07: it needs V0 to V6 | **Closed, 07 is right.** `run` writes `markdown(doc)` (`screen.py:889`), which calls `_brief(doc)` (`:1129`), which reads V0 to V6 [read]. `screen_rung` alone needs none of them |
| X4 | 08 (6.2, row 20) against 05 (6.4, hazard 4) | 08: `env_configs` hold the absolute `run_dir`. 05: the string as given (relative in the committed run) | **Closed, 05 is right.** `base.setdefault("run_dir", str(Path(out) / body))` (`train.py:295`) [read]: relative if `--out` is. A `config={"run_dir": ...}` passed to `train.train` overrides it, so references need not live under `<out>/<body>` |
| X5 | 03 (5.3) against 01 (5.5), 06 (3) | Not a factual clash, a scope gap: 03's "0.40 m/s" thigh-twist table is on `look.speed_spec(H)` (T 0.861 s, duty 0.573). The 75 cm track's 0.40 m/s references are the PR's spec (T 0.993 s, duty 0.55) and `F25_F9_cl` (duty 0.59) | **Closed by [critic run]**, section 3: both saturate both hips at pi/4 |
| X6 | 01-05, 07 against the task's own wording | The maps steer new code to `designlab/stage2/` (no hash moves). `NEXT_SESSION_PROMPT.md:118` says "behind options that default to today's behaviour, keeping G2 bit-identity"; `STAGE2_PLAN.md:138` says `train.py` is "extended with R5 and R6" | **Open (lead's decision).** Facts both ways are in the maps: every committed panel record already reads as made by other code (04 section 0.1); `train.py` refuses a resume on a moved sources hash unless flagged (05 section 6.5) |
| X7 | 02 (2.8) against 03 (14), 06 (9) | 03 and 06 list the two-step stop as "not found". 02 shows the existing stop already has one touchdown after its command at every duty, so the bar's "at most two touchdowns" is met by the reference as built | **Closed as a reading.** What the build must change is the deceleration, not the step count: peak path deceleration `1.875 v / (0.4 T)` is 2.18 m/s^2 on the V0 0.40 m/s walk (02) and 1.89 m/s^2 on the F25 walk (T 0.9934 s) [arithmetic], against the bar's 1.5 (1 Hz filtered, so not the same quantity). A ramp of at least 0.50 stride brings the F25 walk's path value to 1.5 [arithmetic] |

## 2. Per item: questions closed here, and what remains

### A. Transitions for any GaitSpec

Closed:
1. **Which spec.** `NEXT_SESSION_PROMPT.md:129-131`: `_tuned_f9_spec` (`stage2/jetson_plem/jetson_plem_study.py:369-390`)
   and `F25_F9_cl`, "at the nominal speed and at 0.30-0.50 m/s", with "starts, changes and two-step stops from the
   official builders". Within this family the crouch is one value (0.05 H at 75 cm), so 02's T15 (one LIPM frequency
   per transition) does not bite.
2. **What a new stop needs to be read.** A stop event is a dict `kind, name, t_ref, command_phase, speed_m_s,
   period_s` (`control.py:1000-1001`); an episode copies `track.events` at its start (`control.py:2325`) and
   `_stop_result` reads `name, t_ref, speed_m_s, period_s` (`control.py:2246-2260`) [read]. So code outside Stage 1 can
   append the event after `control.build_track`. `tune.build_spec` cannot be used for it: it builds its own track
   (`tune.py:479`) and raises unless there is exactly one stop, or none on a custom walk plan (`tune.py:487-488`).
3. **What records a new file needs.** `tune.read_inputs` needs only `file` and `sha256` per entry, under
   `references` or `transitions` (`tune.py:320-347`, `:333`). `build_spec` then reads
   `transitions[<first segment>]["gaits"]["b"]["duty"]` (`:484`), `references[<gait>]["period_s"]` and
   `["speed_m_s"]` (`:490-497`), and for turns `["yaw_rate_rad_s"]` (`:501`) [read].

Remains open:
- Look fields in a transition (needed for V6, the frozen reference): which features a start, change and stop must
  carry, and how a K-stride F14 window joins (`initial_state` reads one stride back through `ref_a.stride_motion`;
  `compare` slices 160 frames). No map covers `gait._generate_look` at the depth needed to port it.
- The "change to 0.40 m/s": from which walk in each family, and whether a one-stride ramp from the F25 nominal walk
  (T 1.55 s, duty 0.58) to 0.40 m/s (T 0.99 s, duty 0.55) passes G3. Untested.
- The two-step stop's own rule (ramp width, command phase), to be fixed before building (X7).
- Where the spec-taking entry lives (X6).

### B. Look-aware pi/4 turns and the thigh twist

Closed:
1. **75 cm, the track's own references** [critic run]: at pi/4 rad/s the F25 + F9 and `F25_F9_cl` walks at 0.40 m/s
   pin both hip yaws at 1.00 of +-0.30 rad with 25 and 34 of 160 frames failing IK. At 0.5 rad/s they fit (0.73, 0.77).
   The cause is in `turning.py:459`: `foot_yaw_vs_trunk_amp = |w| duty T / 2` is 0.194 rad on V0's walk, 0.215 and
   0.230 on the two F25 walks, because their stride period is 15 % longer.
2. X2: the finite turn needs its own plan function.

Remains open:
- What the 75 cm turn does about the range: quicker steps (`STAGE2_PLAN.md:207`), a lower rate, or lever 6. A rule to
  set before any turn reference is built; the bar's row (`:197`) has no fallback.
- The finite turn's design: window, ramps of at most 0.64 s (03 section 6), the command instant, left and right as
  separate tracks.
- A look-aware (V6) turn: F5's rolling contact under a yawing foot, a tail-phase check on a turn, F14.
- Full-setting values (6 ZMP passes, grip need) for any pi/4 turn; the 60 cm F25 walks (not run).

### C. New bar readings

Closed:
1. **The definitions** are `STAGE2_PLAN.md:180` (own-swing toe clearance, "lift-off to touchdown", Stage 1's reading
   kept beside it) and `:196-199` (4.0 m along the commanded heading within 0.25 m of that line; heading within 10
   degrees of start + 90 at 2.0 s after the command and for 3 s after; at most two touchdowns by the 2 % load rule
   before the speed stays under 5 %).
2. **A reference-to-Recorder adapter exists**, which no map names in that role: `panels.reference_recorder(p, ref,
   seconds)` (`panels.py:2162`) plays a straight periodic reference kinematically into a `metrics.Recorder`. It
   refuses any other reference (`:2167`), so G9 on turns, stops and starts needs a new one.

Remains open (rules, not code):
- Drift: the line's origin and direction (`rec.col("yaw")[0]`, `Built.home_yaw`, or the reference's path), and
  whether 0.25 m is the largest deviation over the 10 s.
- Turn: the command instant (none exists in code) and which clock "2.0 s" is on.
- Stop: the window in which touchdowns are counted.
- Own swing: `low0/low1` are the lowest point of every foot geom (04); Stage 0's gate reads pads only (07).
- G9's 10 ms turn traces: absent, git-ignored, and in no `restore_inputs.py` group.

### D. F25 rungs in the ladder and the Stage 0 screen

Closed: the specs (A.1) and their speeds, which need speed keys beyond `nominal, v040, fr05`.

Remains open:
- Merge or separate directory. Merged as sign-flipped specs they fail `tests/test_stage2_look.py:692` (01 section
  8.3), `:661` and `test_stage2_motion.py:199`. In a separate directory `screen.run` cannot be used (X3); the
  per-rung route of 07 section 1.7 (a) can.
- The L1 page says twice that F25 is not used (`look_page.py:1083`, `:1892`).
- The nominal-speed F25 rung sits outside the cadence and stride spec by construction (06 section 3). No tail sweep
  exists on an F25 walk.

### E. Official panels on the new references

Closed:
1. The inputs contract (A.3). `evaluate` checks all 26 stock files whatever is selected (`panels.py:1827`), so a new
   inputs directory needs the stock files and entries plus the new ones.
2. `Context.built` takes omega from `refs/nominal.npz` at the default 0.03 H crouch (`panels.py:996`,
   `control.py:943`) [read]; the PR's probe used 0.05 H (06 section 5).

Remains open:
- Which omega an official panel uses on a 0.05 H reference (a rule; about 1.2 %).
- Runner: rebinding `panels` globals or an own runner (04 section 14); conditions must keep a stock judge string.
- Cost: 3 walks x 2 grips x 6 delays x 2 rows x 20 episodes = 1,440 episodes, about 2 core-hours before attribution
  [arithmetic from `STAGE2_PLAN.md` section 6's 0.75 core-h per 28 conditions].

### F. R5 env beyond 14 joints, G8a, G9

Closed:
1. **A look rung can become a tracker reference.** `look.look_inverse_dynamics` returns `tau, omega, cop_margin,
   friction, normal` (`look.py:785`, `:844`), every key `refs.arrays` adds [read].
2. **A K-stride window loads on a `Track`.** `add_periodic` passes `frames_per_stride=ref.n` and `period=ref.period`
   (`control.py:967-978`), and `Reference.period` is `n * step` (`refs.py:361-362`) [read]. Caveat: everything that
   reads `period` as one stride (the env's phase input, `strides`, `build_spec`'s `period_s`) is then K times long.
3. **G9's text** is `STAGE2_PLAN.md:128`; its Stage 1 model is `panels.g7` (`panels.py:2226`) with C.2's adapter.

Remains open: mid-task starts (the root pose at a mid-track time is not in a `Track`); speed and yaw-rate commands
as inputs; the head command; the new reward terms; R5's task list and names; G8a's seed alignment (05 section 11);
where the env lives (X6); `tests/test_stage2_x.py:629-637` must change.

### G. R2L refit with fixed delay draws

Closed:
1. **What "fixed delay draws" means**: `LESSONS_LEARNED.md:293-298`. Stage 1b drew each episode's delay from the
   episode seed alone, one episode per kind per generation, so scores tracked the draws. R2L gets fixed or repeated
   draws per kind and verdicts per delay (`JETSON_PLEM_REVIEW.md:235-237`; `NEXT_SESSION_PROMPT.md:165`).
2. **`speedfit` cannot be re-pointed.** `Evaluator` builds `tune.EPISODES` and the two stock speed walks by name
   (`speedfit.py:102`, `:434-446`) [read]. `Evaluator.episode(..., latency=k)` forces a delay (`:480`); the draw is
   `latency_steps` (`:282-285`), uniform 0-5 steps. R2L needs its own evaluator and fit module.

Remains open: `speedfit.fit` (`:701`), `tune.score`, the CMA settings and `speed_g5` are in no map; R2L's task mix
and objective terms for the bar's tasks; its start gains; and its reference: the plan fits it "without F5's
toe-off" (`:124`), but no rung is V6 without F5 (the ladder adds F4 and F5 together at V3, 01 section 3.3).

### H. Notebook, bundle, resume

Closed: X4.

Remains open (none settled by reading code): the exact text of the four printed lines; key schemas for STATUS,
SESSION, MANIFEST, DONE and HALT; Colab's Python patch, numpy, scipy, torch and vCPUs; which version set R5 trains
under; the cross-version resume flag; whether binaries move through the Drive connector.

## 3. The critic's run (not a record)

`75-M-H-10`, `turning.generate`, left turn, `zmp_iters=0`, n 160, r2, one process. Script:
`/tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/critic/twist_check.py`.
The V0 rows reproduce map 03's.

| 0.40 m/s walk | T, s | duty | Rate, rad/s | `foot_yaw_vs_trunk_amp`, rad | Hip-yaw range use, left / right | IK |
|---|---|---|---|---|---|---|
| V0 `look.speed_spec` | 0.8606 | 0.573 | 0.50 | 0.1233 | 0.569 / 0.626 | ok |
| V0 `look.speed_spec` | 0.8606 | 0.573 | pi/4 | 0.1937 | 0.877 / 1.000 | fails, 6 frames |
| F25 + F9 (PR's spec) | 0.9934 | 0.550 | 0.50 | 0.1366 | 0.652 / 0.732 | ok |
| F25 + F9 (PR's spec) | 0.9934 | 0.550 | pi/4 | 0.2146 | 1.000 / 1.000 | fails, 25 frames |
| `F25_F9_cl` | 0.9934 | 0.590 | 0.50 | 0.1465 | 0.680 / 0.774 | ok |
| `F25_F9_cl` | 0.9934 | 0.590 | pi/4 | 0.2302 | 1.000 / 1.000 | fails, 34 frames |

A pinned joint reads 1.00, never more; the failing-frame count is the measure of how far over it is.

## 4. Subsystems no map covers

- `speedfit.fit`, `tune.score`, the CMA-ES loop and `speed_g5` selection (G).
- `panels.g7` and `reference_recorder` as G9's template (C, F).
- `gait._generate_look` at porting depth, feature by feature (A, B).
- `stage2/research/research_look.md` sections 3-4: F11, F12, the look-at and roar, and the look bands' definitions.
- `LESSONS_LEARNED.md` section 5.3's open rules as one list (whole-step delays, S2-4's reading point, how Colab
  checkpoints are judged here).
