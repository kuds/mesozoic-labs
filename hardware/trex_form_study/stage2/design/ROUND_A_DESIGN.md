# Round A design: reference sets, official builders, the bar's readings and panels

Architect's final design, 14 Nov 2026, for the lead. It replaces the draft of the same name after three independent
reviews (provenance, dynamics, scope); the last section logs every finding. Scratch file, not committed. Paths are
relative to `/home/user/mesozoic-labs/hardware/trex_form_study` unless absolute; code line numbers are of commit
`167dce4` (the code is unchanged since `7e9ea0d`). "R1"-"R8" are the lead's requirements, "D n" this document's
decisions, "ruling n" the lead's rulings of 14 Nov 23:10 UTC (`lead_notes.md`). Nothing in the checkout was edited and
no pytest was run, by me or by the reviewers.

Every number marked **[proto]** comes from a scratch experiment of this session and every number marked **[review]**
from a reviewer's scratch run. Neither is a record and neither may be cited as one. They show that a construction
works and what it costs.

Three places differ from a ruling, each because a reviewer showed the ruled definition has a defect. They are marked
**[amends ruling 5]** where they occur and are the first open questions in section 7.

## 0. What was run, and what it settled

Environment of every command: cwd the study root, `D=/tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/design`,
`PYTHONPATH=.:$D MUJOCO_GL=disable OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 nice -n 10 .venv/bin/python`. Scripts:
`$D/proto_moves.py`, `$D/proto_causal.py`, `$D/exp1`-`exp8` (mine); `$D/review_mech/r1`-`r9`, `$D/review_bitid_*.py`,
`$D/review_cov/c1`, `c2` (the reviewers'). Body `75-M-H-10` throughout (H 0.501854 m); "V0 0.40" is
`look.speed_spec(H)` (T 0.8606 s, duty 0.5732).

| # | Run | Result |
|---|---|---|
| E1 | `exp1_legacy_plan.py 75-M-H-10` | A `plan` that takes its schedule as data returns `transitions.plan`'s arrays exactly (`array_equal` on every array, the out slice, ramps, stand points, swings) for all 18 `TRANSITIONS`. 0.02 s per plan. It passed `transitions.gait_of`'s gaits straight in, so it did not test `GaitX` (finding B6). |
| E2 | `exp2_stop.py 75-M-H-10 V0` | Stage 1's one-step stop built from V0 0.40: path deceleration peak 2.18 m/s^2, CoM 1 Hz low-passed forward deceleration 0.64 m/s^2. Its CoM leaves the continuing walk (> 1 mm) 0.58 s before `stop_command_phase`'s "command". |
| E3 | `exp5_causal_cart.py 75-M-H-10 stop` | Cart table, preview controller that learns of the stop at the command frame: one step, command at the last lift-off, needs the ZMP 69 mm off its target (pad reach 63 mm: not feasible); Stage 1's stop plan commanded at the right foot's lift-off (phase -0.5): 8 mm, two touchdowns after the command. |
| E4 | `exp4_stop_causal.py 75-M-H-10 V0 -0.5 legacy`; [review] `r3_cached.py` | That stop, full solve, 6 corrections: 38 s; `ik_ok`; consistency error 18.9 -> 0.67 mm; causal transient 9.2 mm; join at the command CoM 0.006 mm; join to the stand CoM 0.15 mm, ZMP 0.43 mm; touchdowns 0.371 s and 0.801 s after the command; forward speed below 5 % from 1.033 s = 1.200 strides; 1 Hz deceleration 0.647 m/s^2 (0.691 at the 5 % fast clock), unfiltered 2.16. |
| E5 | `exp5_causal_cart.py 75-M-H-10 left`; [review] `r2_plan.py` | Quick turn, plan level, ramps of half a stride: V0 (f 0.8, ramp 0.3873 s) 80.74 deg at 2.0 s of reference time, 85.17 at the bar's 2.0 s, 80.51 with the clock held 5 % slow. **F25F9cl (f 0.65, ramp 0.4098 s): 79.67, 84.15, 79.45 deg.** Left and right identical to 1e-15. |
| E6 | `exp6_turn.py ... V0 left 0.8:0.5:0.0 full`; [review] `r4_right.py` (the right turn in full) | Causal 90 deg turn from V0 0.40 at pi/4 rad/s, stride x0.8: 77 s for 7 solves; `ik_ok`; hip-yaw use 0.817 on the worse (outside) leg and 0.765 on the other, in the left turn and, mirrored, in the right; consistency 48 -> 1.7 mm; causal transient 27.5 mm; join at the command CoM 0.001 mm; last stride against the walk CoM 0.10 mm, ZMP 1.42 mm, both directions; heading 90.000000 deg. Kinematic passes (10-13 s): x0.85 0.875; x0.9, x1.0 pinned with IK failures; F25F9cl x0.7 0.894, x0.65 0.823, x0.6 0.753. |
| E6b | 14 Nov 23:17, `exp6_turn.py 75-M-H-10 V0 right 0.8:0.5:0.5,0.8:0.497:0.5` and `... F25F9cl right 0.65:0.5:0.5,0.65:0.47:0.5,0.65:0.44:0.5` (kinematic) | The ramp's length hardly moves the thigh twist: V0 0.818 at both; F25F9cl 0.823, 0.822, 0.820 for ramps of 0.410, 0.385, 0.361 s, with the reference heading at the bar's 2.0 s 84.15, 84.76, 85.37 deg. A ramp fixed in seconds costs nothing in range. |
| E7 | `exp7_turn_drives.py 75-M-H-10 V0_turn_left`; [review] `r9_cl_speed.py` | Stage 0 ratios of the V0 turn: hip-pitch speed 0.923, knee speed 0.925, every leg ratio below 1; **tail peak 1.199** (walk: 0.728; before ruling 1's amplitude rule); single-support CoP margin 24 mm (walk 45 mm); single-support grip need 0.90 by `turning`'s torsion formula (walk 0.63). [review] F25F9cl turn at x0.65, kinematic pass: hip-pitch speed 0.968, knee 0.861, twist 0.823. [review] the V0 turn reference is free of leg interference. |
| E8 | `exp8_change.py 75-M-H-10 V0`; [review] `r7_change_cl.py` | Direct change nominal -> 0.40 m/s, Stage 1's construction: V0 G3 CoM 0.29 mm, ZMP 0.58 mm; **F25F9cl (T 1.551 -> 0.993 s) CoM 0.25 mm, ZMP 0.96 mm, `ik_ok`.** |
| V1 | [review] `r5_zmp.py`, `r6_taper.py` | In the V0 right turn the excused "causal" ZMP term is 27.6 mm in the first half stride, 6.0 in the second, then 2.0-3.7 mm at every lift-off for the rest of the move (the walk itself: 1.85 mm): after one stride it is the preview controller's ordinary error. Fading it to zero over phases [command + 0.5, command + 1.0] gives join ZMP **0.727 mm** (from 1.42), CoM 0.105 mm, twist unchanged. |
| V2 | [review] `review_bitid_accuracy.py` | After `look.generate` the process is back at accuracy r1 while `info["accuracy"]` says r2; `gait.inverse_dynamics` taken then differs from r2 on frames 0, 1, 158, 159 (tau up to 2.33 N m) and the file hashes to `64a7dcab...`. At r2 the same arrays, written through a file object, hash to Stage 1's `b767df97...`. |
| V3 | [review] `review_bitid_lookgate.py` | `look.arrays(src, look.generate(src, spec))` for `sizing.NOMINAL` and `look.speed_spec(H)` hash to the committed look ladder's V0 rungs (`e2bf5274...`, `4545bad6...`). |
| V4 | [review] `python -m designlab.stage1 eval --body 75-M-H-10 --rows R0,R2/42 --conditions start_stop/twice,walk/brisk --episodes 1 --workers 1 --inputs stage1/runs/2026-11-08 --out <scratch>` | All four traces equal the sha256 in the committed 8 Nov episode records, although the panels' source hash has moved since. Under 3 min. |
| V5 | [review] `c1_records.py` (reads committed records) | `75-M-H-10` `walk/brisk` (0.351 m/s) and `walk/fr05` under R2/42, R2/43 and R0: 20 of 20 up, **0 of 20 succeed, `drive_sustained` failing in 20 of 20 with `tail_yaw_servo` the worst drive** (median 1.048; 1.18-1.27 at Fr 0.05). |
| V6 | [review] `r8_warp.py` | With V6's F14 values on the 0.40 m/s walk (K = 12 strides, 10.33 s), lift-offs fall up to 8.86 ms (1.65 frames, 3.55 mm of CoM travel) off the uniform schedule; only the window boundary is exact. |
| V7 | [review] `c2_checks.py` | On update rows alone the own-swing clearance over-reads by up to 5.6 % of the lift as the lift-off moves across the 10 ms grid; plan lengths: V0 episodes 18.9-21.1 s, F25 sets 23.5-24.4 s; R2 costs 0.39-0.47 s of wall time per simulated second, R0 0.08. |
| - | read `stage1/runs/2026-11-08/*/eval/*/turn/walking_0.50_*.json` | Stage 1's 0.5 rad/s turn references use 0.847 of the thigh twist on `75-M-H-10`; R2/42 and R2/43 kept decision 14's range in 20 of 20 there (0 ms). On `60-N-T-10` (0.766) the range term failed in 1-6 of 20. |
| - | `git check-ignore -v` on planned paths | `*.npz` (root `.gitignore:49`) and `logs/` (`:34`) are ignored at any depth; `*.tmp`, `partial/`, a lock file and traces' directories under `stage2/sets` or `stage2/runs` are **not** until the two new `.gitignore` files exist. |

What these settle:

1. **The stop's problem is the command instant, not decision 10.** The one-step stop's reference already reads
   0.64 m/s^2 at 1 Hz. Stage 1's references are made with unlimited foreknowledge, so the CoM departs half a second
   before the "command". An honest "at most two touchdowns after the command" needs a reference that is the walk,
   exactly, up to the command (D5; ruling 2 accepts it).
2. **Such a reference is buildable** with Stage 1's preview controller, IK and correction loop (E4, E6), in both turn
   directions, with the join error halved once the excused transient is bounded to one stride (V1).
3. **The quick turn fits the thigh twist only with quicker steps** (x0.8 for V0, x0.65 for F25F9cl), and a ramp
   counted in strides gives the candidate a slower ramp than the control (E5). A ramp fixed in seconds removes that at
   no cost in range (E6b).
4. **On this body the tail servo alone decides the "every limit" reading at 0.35 m/s and above** (V5). The bar's
   records must therefore carry the plan's third reading ("shown but for A1") from the first episode on.
5. **Walk files depend on process state** unless the builder fixes it (V2), and the regression available today is
   wider than the 160 turn traces (V4).

## 1. Decisions

**D1. Panel integration: an own runner on the panels' classes (option iii), in `designlab/stage2/bar.py`** (ruling 6).
Rejected: (i) hook edits to `panels.py`: a walk on another gait, a new judge, a second seed block, two success
readings and per-condition input files need changes at nine dispatch points (`conditions` `:453`, `template` `:519`,
`files_of` `:550`, `Built.legs` `:926`, `terms_of` `:1031`, `judge` `:1099`, `counts` `:1390`,
`PanelWorker.__init__` `:1771`, `summarize` `:1965`) inside the panels', the tuner's and the speed fit's source
hashes. Rejected: (ii) rebinding `panels` attributes at import: it changes `panels` for everything else in the process,
and a spawned worker checks its inputs with the stock `conditions()` before the rebinding module is imported.
Why (iii):
- *Provenance.* The runner hashes the modules an episode executes (`bar_sources`: `bar.py`, `readings.py`,
  `setio.py`, `speedfit.py`) beside `panels.panel_sources()` and `panels.key_context(body)`, as `ledger.py:931-973`
  does. Round A changes one Stage 1 line set (D18), before the freeze, so every Stage 2 record carries one panels
  hash.
- *Multi-worker safety.* The worker factory is a top-level function of the importable module
  `designlab.stage2.bar`; the CLI guard at the bottom of the file imports that module by name and calls its `main`,
  so no class ever lives in `__main__` (finding B11). A spawned worker imports Stage 2's conditions before it reads
  an input, verifies the same `bar_sources` and thread settings, and rebuilds every object from plain data.
- *Later rows.* The runner drives any `panels.Row` through `row.episode(ctx, cond, seed, relax)` and judges with
  Stage 2's judge. The servo delay is a wrapper around any row (`bar.Delayed`), so R5's policy rows
  (`policy_rows.PolicyOnlyRow`, a `panels.Row`, not a `TrackerRow`) and Colab-trained checkpoints get a delay and a
  bar record without a new class.
- *Stage 1's twelve "no loss" conditions* run through the same runner: `bar run --set S1 --stage1-run
  stage1/runs/2026-11-08` reads a Stage 1 run directory as the pseudo-set `S1`, builds each stock condition with
  `panels.Context.built` and judges it with `panels.judge` unchanged. Records go to the bar's own tree with
  `definition.stock = true`. The stock `eval` is not used for them: its parser cannot take a delayed `R2/42` row
  (`panels.py:887-888`), and it overwrites on a key change.

**D2. A reference set is a directory `stage2/sets/<set>/<body>/` in Stage 1's run-directory layout.**

```
stage2/sets/.gitignore                    *.npz  *.tmp  .lock  partial/  traces/  logs/  done/   (committed)
stage2/sets/legacy_check.json                      the 36-transition bit-identity record (D4)     committed
stage2/sets/<set>/<body>/refs/<name>.npz           Stage 1 reference format (refs.arrays' keys)   git-ignored
stage2/sets/<set>/<body>/transitions/<name>.npz    Stage 1 transition format (+ look keys later)  git-ignored
stage2/sets/<set>/<body>/look/<name>.npz           look.arrays' format, periodic walks only       git-ignored
stage2/sets/<set>/<body>/refs.json                 Stage 1's schema + Stage 2 keys                committed
stage2/sets/<set>/<body>/transitions.json          Stage 1's schema + Stage 2 keys                committed
stage2/sets/<set>/<body>/set.json                  definition, rules, look entries, freezes       committed
stage2/sets/<set>/<body>/screen.json, SCREEN.md    Stage 0 screen of every reference              committed
stage2/sets/<set>/<body>/partial/<name>.json       one finished item not yet merged (resume)      git-ignored
```

Names are made and parsed by `setio` only, numbers with `%g`: walks `nominal`, `v0.3`, `v0.35`, `v0.4`, `v0.45`,
`v0.5`; start `start`; changes `nominal_to_v0.4` and `nominal_to_v<v>` (D21 names the fallbacks); stops `stop2_v0.4`
(the bar's) and `stop1_v0.4` (Stage 1's one-step stop, a labelled comparison); turns `turn90_<rate>_<left|right>`
with the rate label from the table `setio.TURN_RATES = {"0.35": 0.35, "0.5": 0.5, "pi4": math.pi / 4, "1": 1.0}`:
the quick turn is `turn90_pi4_left` and `turn90_pi4_right`. No name holds a rounded value: a test maps every name
back to its value exactly. No transition of a set may be named `stop` or `stop_in_place_*`: `control.Track`
registers Stage 1's stop event by those names (`control.py:994`), and the bar registers its own (D5). Sets: `V0`,
`F25F9`, `F25F9cl` now; `V6`, `V6noF5` later. Every default path is anchored at the study directory
(`s2io.STUDY`), never at the working directory.
Stage 1's `Planner`, `Context`, `control.build_track`, `Reference.load`, `Transition.load` and
`tune.build_spec(plan=, gait=)` load such a directory unchanged ([review] confirmed: `tune.py:479-497`,
`control.py:1099-1132`, fractional periodic strides, mid-stride finite files).
No gains are copied into a set: rows read `stage1/runs/2026-11-08/<body>/tune.json` through `panels.r2_gains` and
record its path (made portable) and sha256. Rejected: writing the new arrays into `stage1/runs/2026-11-08/<body>/`:
it would change files every Stage 1 record key hashes whole.

**D3. The three sets on `75-M-H-10`.** Speeds `nominal` (Fr 0.012: 0.2431 m/s) and 0.30, 0.35, 0.40, 0.45, 0.50.
- `V0` (control): `sizing.NOMINAL` at the nominal speed; between two of Stage 1's walks `look.speed_spec(H, v, lo,
  hi)` with (NOMINAL, BRISK) below BRISK's speed and (BRISK, FAST) above it; above FAST's speed
  `replace(envelope.FAST, froude=v*v/(G*H))`, labelled `new` (ruling 5). On 75 cm only 0.50 m/s is above FAST.
- `F25F9` (control): `_tuned_f9_spec(body, H, v)` copied verbatim into `sets.tuned_f9_spec`; a test imports the PR's
  script by path and compares the two functions field by field at every rung on both bodies.
- `F25F9cl` (candidate): `replace(tuned_f9_spec(body, H, v), duty=0.59, lift_frac=0.095)` at every speed. Rungs
  other than 0.40, 0.45 and 0.50 m/s carry `label: "new"`; every entry carries `margin_rule: "duty and lift set
  after results on seeds 3042-3061 (JPS 2.3)"`.
- The crouch is read from the spec, per body (the PR's spec gives 0.05 H at 75 cm and 0.07 H at 60 cm). `sets
  verify` asserts that every spec of a set and body has one crouch and records it in `set.json`; the builder refuses
  a move between two crouches (`transitions.py:643-644` uses one LIPM frequency per transition).
- **Process-state contract.** Every array-producing call runs inside `s2io.r2()`, which sets
  `gait.set_accuracy("r2")`, asserts `gait.ACCURACY == "r2"` on entry and on exit of the item and restores the
  previous level. A walk is `g = look.generate(src, spec)`, then `gait.inverse_dynamics` and `gait.summarize` and
  `refs.arrays` **inside the same block** (V2). Each entry records `built_by.accuracy`. No item relies on a level an
  earlier item left behind.
- Built-in gates of the builder: V0's `refs/nominal.npz` and `transitions/start.npz` hash to Stage 1's own
  (`stage1/runs/2026-11-08/<body>/refs.json`, `transitions.json`); V0's `look/nominal.npz` and `look/v0.4.npz` hash
  to the committed look ladder's V0 rungs (`stage2/look/look_refs.json`), and their screen blocks equal
  `stage2/look/screen.json`'s (V3; D15); for every walk of every set, the Stage 1-format file's `q`, `qd` and `qdd`
  are `array_equal` to the look-format file's (this catches a walk whose inverse dynamics ran at r1).

**D4. One generalised builder, `designlab/stage2/moves.py`; Stage 1's transitions are one of its schedules.**
`moves.plan` is `transitions.plan` with the schedule passed in as data (E1). `moves.generate` runs the same loop
with Stage 1's own `com_path`, `solve_frames`, `derivatives`, `multibody_zmp`, `idyn`, `initial_state`, `arrays`,
`compare` and `summary` when the move is non-causal and has today's laws, so that path executes the same
floating-point operations in the same order. `GaitX` carries `in_place` so that `legacy()` is `gait_of`'s two
constructor calls argument for argument (4 of the 18 legacy transitions are in-place turns).
The round's central bit-identity claim is kept as a committed record, `stage2/sets/legacy_check.json`: all 18
Stage 1 transitions of both picks regenerated through `moves` at the full passes, each npz's sha256 against
`stage1/runs/2026-11-08/<body>/transitions.json`, with the sha256 of `moves.py`, the panels' source hash, versions,
thread count and commit. `sets freeze` requires a passing record made by the current `moves.py`; a test asserts the
same. Rejected: options added to `transitions.plan` and `generate` (the same result with four Stage 1 hashes moved);
hand-assembled calls per reference (the PR's start).

**D5. A commanded move is causal, and its file begins at its command instant** (ruling 2). For a stop and a turn,
the reference is the periodic walk, frame for frame, up to the command frame; the preview controller first sees
the commanded plan at that frame; the move's file starts there, and the track plays the periodic walk up to it. The
command instant is therefore `segment.t0` of the move on the track. In round A a command may only be given at a
lift-off (a multiple of half a stride): at any other phase a swing is in the air whose landing would already head
for the commanded path, so the frames before the command would differ from the walk; the builder refuses it.
Rejected and named in the pre-registration: Stage 1's convention (command = start of the reference's ramp, decision
18), whose CoM anticipates by 0.4-0.6 s; command = first measured departure beyond a tolerance. Decision 18's
reading stays in every stop record. Starts and speed changes stay Stage 1's construction (non-causal): no bar
reading is tied to their command. `setio.add_events` appends one event per causal move of a plan to the track
(`kind`, `name`, `t_ref`, `command_phase`, `speed_m_s`, `period_s`): the bar, G9 and R5's env share it.

**D6. The quick turn** is one causal move `turn90_pi4_<dir>` from the set's `v0.4` walk to the same walk on the
heading + 90 deg (section 3.4). Yaw rate and stride period ramp together from the command, hold, and ramp back;
path speed, duty and lift stay the walk's; the command is the lift-off of the inside foot (left turn: a left
lift-off; right turn: a right lift-off), so the two directions are mirror plans.
- **The ramps last 0.375 s of reference time, for every set, body and rate** **[amends ruling 5, "turn ramps are
  half a stride"]**. In strides that is `r = 0.75 / (T_walk + T_turn)`: 0.484 for V0 (f 0.8), 0.458 for F25F9cl
  (f 0.65). Why: a ramp counted in strides is 0.387 s for V0 and 0.410 s for the candidate, which leaves the
  candidate's reference at 79.7 deg at 2.0 s of its own clock and one degree behind the control at the judged
  instant (E5); 0.375 s puts both at or above 80.3 deg even with the clock held 5 % slow (arithmetic from E5 and
  E6b: the heading at the judged instant moves 24.8 deg per second of ramp), and the twist does not notice (E6b).
  The value was set on plan-level arithmetic of both sets before any closed-loop run.
- **Ruling 1's appendage rule.** Inside the turn the tail's and the neck's swing amplitudes are
  `amp x (T(p) / T_walk)^2` (constant angular acceleration amplitude), for every set, look sets included; it is a
  recorded law of the move (`rules` holds `S2A-appendage`), labelled a look choice.
- Rejected: outside foot first (hip-yaw use 0.929 against 0.819, E6); `enter_* + ref + exit_*` (at least 232 deg).

**D7. The stride rule (fixed before any closed-loop run).** Inside the turn the stride period is `f x T_walk` at the
same path speed. `f` is the largest value on the grid 1.00, 0.95, ..., 0.60 for which the finite turn's own
kinematic pass (the causal plan solved once, no ZMP correction, both directions) has `ik_ok` and a hip-yaw use of
at most **0.85** of the usable range on the worse leg. The reference is then built with all passes and must show
`ik_ok` and at most **0.87**; if not, `f` steps down once (at most twice in all) and the reference is rebuilt. The
steady turn's kinematic reading is recorded beside it. Outcomes, by rulings 1 and 4:
- no `f >= 0.60` passes, or the built reference is over a **leg** drive's Stage 0 limit: **no reference is frozen
  for that set, body and rate**; the set's record says so with the numbers, the bar's turn item gets a "not run"
  record reading "no reference within the thigh twist and the leg drives", and it goes to the owner (lever 6, or
  lever 1 where speed is the limit). Nothing is designed around it.
- the built reference is over a **neck or tail** planning limit after the appendage rule: it is frozen with
  `a1_over: true` and read under the plan's third reading only.
Why 0.85: decision 14 asks the closed loop to keep the reference's own margin less 0.02 rad; at 0.85 the reference
keeps 0.045 rad, the value at which R2/42 and R2/43 held the range in 20 of 20 on this body at 0 ms (Stage 1's
0.5 rad/s turns, 0.847). The constants 0.85, 0.87, the grid and the step-down count were set on the 14 Nov
prototypes of V0 on `75-M-H-10`; the rule's text says so. [proto] V0 picks 0.80, F25F9cl 0.65. The other ladder
rates (0.35, 0.5, 1.0 rad/s) are the same move with another rate and the same rule.

**D8. The two-touchdown stop** `stop2_v0.4` (ruling 3) is Stage 1's stop plan (`_param_fns("stop")`: left lifts
last at phase 0 and lands beside the right; speed ramp over `[phi_c - 0.2, phi_c + 0.2]`) built causally with the
command at the right foot's lift-off, phase -0.5 (section 3.5). After the command the right foot lands on its
periodic footprint, then the left: two touchdowns; [proto] 1.20 strides to 5 %, 0.65 m/s^2 at 1 Hz. Rejected: a
speed ramp from the command with a shortened right step (31 mm causal transient against 8 mm); a one-step causal
stop (69 mm: off the pad). `stop1_v0.4` is Stage 1's one-step stop on the same gait, non-causal, built and screened
as a labelled comparison.

**D9. Clocks.** Three, named in every record: the *reference clock* `t_ref` (track time; `metrics.COLS[1]`);
*simulated time* `t` (`COLS[0]`); the *bar's clock* `tau = t - t_c`, simulated seconds since the command instant
`t_c = rec.sim_time(t_ref_cmd)`, `t_ref_cmd` the track time of the move's first frame. The panels play the track
at `RATE = 1.05`, times the tracker's phase-rate factor (within +-5 %, `control.PHASE_RATE_LIMIT`). "The reference
itself at bar time tau" means the reference at `t_ref_cmd + RATE x tau`, the nominal commanded clock; "at the
slowest clock" means at `t_ref_cmd + 0.95 x RATE x tau`. A turn at rate w is read at bar time
`at_s = (pi/2) / |w|` (2.0 s exactly for the quick turn; ruling 5) and held for 3 s after. G9's rule for a turn
reference: heading at bar time `at_s` at least 80 deg on the nominal clock, and at least 80 deg at the slowest
clock (the margin, recorded with its value); the value at `at_s` of reference time is recorded too. All judged
times of the plan's section 5 (2.0 s, 3 s, 10 s) are bar-clock or simulated seconds, never reference seconds.

**D10. Readings carry what they measure in their names, and every episode is read three ways.** Terms:
`toe_clearance_own_swing` (judged) beside `stage1.toe_clearance` (Stage 1's, on the reference's clock); `along` and
`line`; `heading`; `touchdowns_after_command`, `stopped_by_end`, `stop_acceleration_1hz`. Each episode has
`success` (the plan's reading 2, every limit) and `success_but_A1` (reading 3: the same terms with
`drive_sustained` read on the leg drives only), each with its own counts, bound, pass and verdict in the record;
reading 1 ("shown with the look") has a reserved block that is `null` with the reason "not defined: the set has no
look bands" on the three non-look sets. Reports give the readings in the plan's order and never pool them.

**D11. Rows.** `R2/42@<ms>ms`, `R2/43@<ms>ms`: Stage 1's fitted controllers unchanged (`panels.r2_gains` on
`stage1/runs/2026-11-08/<body>`), inside `bar.Delayed(row, k)`, which runs the inner row's `episode` within
`speedfit.servo_delay(b.p, k)`, k = ms / 2 physics steps, 0 <= k <= 10. `R0@<ms>ms`: the panels' R0
(`TrackerRow(mode "R1", terms=(), ankle_cap=())`) in the same wrapper; `R0@10ms` is Stage 2's R0L at the judged
delay. Round A runs k = 0..5. With k = 0 the wrapper changes nothing (tested bit for bit). `Delayed.sources()` is
the inner row's plus `bar.py` and `speedfit.py`. These names are not the speed fit's `R2@10ms` (its body-level
gains) nor its `R2s/42@10ms`.

**D12. Conditions** (16 gated per set and body, as the plan's table): `walk/v0.4`, `turn/q90_left`,
`turn/q90_right`, `nudge/forward|backward|left|right`, `stop/two_step`, each also with `_mu0.54`. Reported:
`walk/nominal`, `walk/v0.3` ... `walk/v0.5` (speed ladder; `_mu0.54` where the run plan asks),
`turn/t90_<rate>_<dir>`, `nudge/<dir>_x0.375`, `_x0.5`, `stop/one_step`. Delay belongs to the row, friction to the
condition, the set and the seed block to the directory.
- **Shared lead-in (a rule, with its reason).** Every item starts with the walk item's lead-in (`start`,
  `nominal` x 1, `nominal_to_v0.4`, then `v0.4`) and takes the walk's seed slot (slot 0), so every item meets the
  robot in the state the walk item judged. The 16 conditions are therefore not independent: a fall in the lead-in
  shows in all eight items of a seed. Every episode records `command_s`, `fell_before_command` and
  `segment_reached`; reports count falls before and after the command apart.
- **Left out of round A, by name:** the turn and nudges from the nominal walk, carpet walks, stiffness x0.7 and
  `robot_floor_carpet` (the plan's reported items; they need the judged controller or other plants and wait for
  Track C); delays of 14, 16 and 20 ms (the wrapper takes them; no run is planned); the look gates and the head's
  cost. The turn-rate and nudge ladders are defined here and run only if task 4's budget allows (section 6).

**D13. Seeds, verdicts, replication, attribution.** Block `seeds3042` (3042-3061, the headline) and block
`seeds5042` (5042-5061, labelled replication). Per record `PASS_RULE` (20 up, at least 17 succeed) on the block's
own 20 seeds, with `panels.lcb`'s bounds, for each reading; per controller `shown` if both R2 rows pass,
`provisional` if one. Judged at 10 ms; every other delay reported, never read between delays.
- *Replication.* "Passed" means one record: a row, condition and delay of set F25F9cl whose `seeds3042` record has
  `pass_` true in reading 2 or in reading 3. Each such record is re-run on `seeds5042` with the same row, and the
  replication record is labelled with the reading that triggered it. Reports print the two blocks side by side and
  never pool them. Order and budget: section 6.
- *Attribution* (decision 20; `panels.attribution_plan`, `attribution_record`, `gates.attribute`): on R2 rows'
  records with 5 or more falls. A turn, nudge or stop record with fewer than 5 falls after its command is not
  attributed itself: it cites the walk record of the same set, row, grip and block ("lead-in falls: see
  walk/v0.4"). Attribution runs on the judged 10 ms records first; a record left unattributed for budget is
  labelled "not attributed (budget, pre-registered order)".

**D14. Identity is separate from provenance, and a finished record is never overwritten.**
- *A bar record's key* holds what an episode executes and nothing else: the row's portable description and
  sources, the condition, the plan, the seconds, the seeds, the command, `panels.key_context(body)`,
  `{panels: panel_sources, bar: bar_sources}` and, per input file, its npz sha256 and the sha256 of a **named
  subset** of its entry (the numbers `tune.build_spec`, `BarPlanner` and `add_events` read: section 3.6). A label,
  a rule text, `look_sha256`, the builder's hashes or another entry added to the set changes no key. `sets.py`,
  `moves.py`, `setscreen.py`, `s2io.py` and `barreport.py` are in no key.
- *A set entry's identity* is its definition (specs, every builder argument, body model sha256, input reference
  sha256) plus its file's sha256. What built it is provenance, recorded per entry (`built_by`) and never
  recomputed: after a no-op edit to `moves.py`, `sets verify` reports "built by other code, arrays identical" and
  nothing else changes.
- *No overwrite.* `bar run` meeting a status-done record whose key differs stops with the differing key fields
  printed. The choices are a new `--out` or `--supersede`, which moves the old record to
  `<record>.superseded/<first 12 hex of its key hash>.json` (kept, committed) before running. Stage 1's "rerun and
  write over" (`panels.py:1862-1888`) is not copied.
- Rejected: a second "judge key" so that a reading fix re-judges from traces: the recorder a re-judge needs
  (500 Hz drive forces, 100 Hz rows, poses) is about 0.5 MB per episode, 5 GB for the two tasks against 4 GB free.
  Instead traces carry the 500 Hz contact bits and both feet's 100 Hz low points, so the own-swing clearance, the
  line and the touchdown count can be re-read, and episodes store per-swing and per-drive tables.

**D15. Every reference gets a Stage 0 screen block in the set's own `screen.json`** (section 3.7), made by
`designlab/stage2/setscreen.py` from `screen.py`'s and `look.py`'s functions. Nothing is written under
`stage2/look/`; `look_refs.json`, `screen.json`, `metrics.json` and their hashes stay as committed. One table says
which gates block a freeze and which are reported. Rejected: adding F25 rungs to `look.ladder` / `look_refs.json`
(it rewrites the file, breaks the whole-file hashes two records hold, and trips five tests).

**D16. The track's LIPM frequency is the set's own** (a pre-registered rule inside "controllers unchanged"):
`control.omega0(p, q_home, crouch_frac=<the set's crouch>)`. Stage 1's call sites use 0.03 for every reference
(`panels.py:996`) and R2's gains were fitted at 0.03; the F25 sets crouch 0.05 H on this body (about 1.2 % in
omega). Every bar record holds `omega` and `crouch_frac`.

**D17. The look (R6)**: section 3.9. `moves.plan` takes its laws from a `Laws` object with fixed signatures;
today's laws are the default and reproduce Stage 1; the look's laws, frame solver, inverse dynamics and K-stride
joins live in a second module that `moves.generate` loads by name, so the look phase edits no file of the non-look
phase.

**D18. One Stage 1 edit, landed before code state A.** `control.Track.add_periodic` reads an optional
`window_strides` key (section 2.3); with no such key the path is bit-identical (`x / 1`, `n // 1`). It is needed
only by the look's K-stride walks, but it changes the panels', the tuner's and the speed fit's hashes and
`plant.sources_sha256` (four), so it lands before any Stage 2 record exists: tasks 3-4 and the later R2L and R5
rows then share one `key.sources.panels`. Its acceptance is the regression of section 4.2. By ruling 8 the Jetson
body's merge also moves these hashes before the freeze; the regression record is therefore (re)made at the freeze
commit, and its test fails until it is. If a later round must edit a module of `panel_sources`, the baseline rows
of any comparison are run again into a new `--out`, never into a state-A directory, and that is pre-registered.

**D19. Engineering.**
- *Threads.* Each CLI module sets the three BLAS variables to 1 at its very top under `if __name__ == "__main__"`,
  before numpy is imported, then calls `s2io.require_one_thread()`, which refuses any value other than "1"; every
  spawned worker calls it on start; every entry and every bar session records `blas_threads`; `sets verify` gates
  on 1.
- *Atomic writes.* JSON through `<name>.json.tmp`, arrays through an open file object on `<name>.npz.tmp` (never
  `np.savez(path)`: numpy would append `.npz`), then `os.replace`; stale `*.tmp` files are swept at start under the
  lock; `*.tmp` is git-ignored.
- *Locks.* One `.lock` per set directory and per run directory (`O_EXCL`; pid, host, start time; stale when the pid
  is gone on the same host).
- *Resume units.* Sets: an item (one walk or one move); the committed JSON is the only truth for a finished item,
  and a `partial/<name>.json` is merged into it and deleted in the same step. Bar: an episode; each finished
  episode is written by its worker to `partial/.../<key hash>/<seed>.json` with its trace's sha256; the condition
  record is written when its 20 are there and the partials are deleted; the record lists its sessions.
- *Processes.* `--procs` (default 2), one BLAS thread, `tools/lowprio.sh`; long runs are one idempotent chain with
  done-markers, one command per delay tier, and a watchdog entry (section 6).

**D20. Freeze.** `sets freeze` appends to `set.json`'s `freezes[]` a manifest of every entry `(name, kind, sha256,
entry_sha256)` once the set's screen is complete and its gates pass. The manifest is append-only by `(name,
sha256)`: a freeze refuses to drop or replace a frozen entry; a changed reference needs a new name or a new set
version (`F25F9cl.2`). On a frozen set `sets build` never rewrites a frozen entry of `refs.json`,
`transitions.json` or `set.json`; it only puts an array in place when its sha256 equals the manifest's, and fails
loudly otherwise. `bar` refuses a set that is not frozen or whose files differ from the latest manifest; records
carry the manifest sha256 of the freeze they ran on; `barreport` checks every record's input hashes are still
listed. `freeze` refuses entries built from differing source hashes or a dirty tree unless the manifest says so
(`mixed_sources`, `dirty`), and requires a passing `legacy_check.json` of the current `moves.py`. State A of the
code freeze = the commit at which the three 75 cm sets are frozen and the regression record is current.

**D21. Fallbacks for moves other than turns, fixed before building.** A start, change or stop that fails its
gates (section 3.3) or a blocking Stage 0 gate (section 3.7) is retried in this order and no other: (1) a change
with a ramp of 2 strides instead of 1 (same name; the entry records `ramp_strides: 2` and the rule); (2) for a
ladder rung, a change from the neighbouring rung toward 0.40 m/s (`v0.4_to_v0.45`, `v0.45_to_v0.5`,
`v0.4_to_v0.35`, `v0.35_to_v0.3`), the rung's plan then walking through 0.40 m/s; (3) the item is recorded "no
transition within the gates" with its numbers. A start and the stop have no fallback. Conditions that need an
absent reference get a "not run" record with the reason, so "all 16 gated conditions have records" stays checkable.
A failed `start` or `nominal_to_v0.4` blocks the whole set and goes to the lead.

**D22. Budgets are kept by a ledger, and the run order is fixed** (section 6): `bar run --task N` adds each
episode's wall seconds to `budget.json` in the run directory; a tier starts only if the task's spent core-hours
plus the tier's dry-run estimate fit the task's 12; a started tier always finishes; what was not started is
reported as cut "by the pre-registered order".

## 2. Module layout

New files (all under `designlab/stage2/`, `tests/`, `stage2/`):

| File | Purpose | In a bar key |
|---|---|---|
| `designlab/stage2/s2io.py` | threads check, r2 context, canonical JSON and sha256, atomic writes, locks, `built_by` provenance | no |
| `designlab/stage2/setio.py` | names; reading a set (entries, manifests, the entry subsets); command events; the track of a plan | **yes** |
| `designlab/stage2/readings.py` | the bar's readings as pure functions of arrays (numpy and scipy only) | **yes** |
| `designlab/stage2/moves.py` | the generalised builder of starts, changes, stops and finite turns for any `GaitSpec`; the legacy check | no |
| `designlab/stage2/sets.py` | set definitions, building, verifying, freezing a set directory; CLI | no |
| `designlab/stage2/setscreen.py` | Stage 0 screen blocks of the walks and moves of a set; CLI | no |
| `designlab/stage2/bar.py` | conditions, plans, rows, judge, runner, records, the budget ledger; CLI | **yes** |
| `designlab/stage2/barreport.py` | machine-made tables from the bar's records | no |
| `designlab/stage2/regress.py` | the closed-loop bit-identity regression against committed Stage 1 and 1b episode records; CLI | no |
| `designlab/stage2/moves_look.py` | later: the look's laws, frame solver, inverse dynamics, K-stride joins (R6) | no |
| `tests/test_stage2_s2io.py`, `_setio.py`, `_readings.py`, `_moves.py`, `_sets.py`, `_setscreen.py`, `_bar.py`, `_barreport.py`, `_regress.py` | section 4 | |
| `stage2/sets/.gitignore`, `stage2/runs/.gitignore` | `*.npz`, `*.tmp`, `.lock`, `partial/`, `traces/`, `logs/`, `done/` | |
| `stage2/sets/legacy_check.json`, `stage2/regress/regress.json` | the two bit-identity records (D4, D18) | |
| `stage2/runs/<run>/run_75cm.sh` | the idempotent chain of section 6 | |

Every module that has a CLI starts and ends like this, so that no class lives in `__main__` and the thread
variables are set before numpy loads:

```python
from __future__ import annotations
import os, sys
if __name__ == "__main__":                       # before any numpy import, as look.py:119-123
    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(_v, "1")
...                                               # the module
if __name__ == "__main__":
    from designlab.stage2 import bar as _named    # the importable copy owns every class and all module state
    sys.exit(_named.main())
```

`main()` calls `s2io.require_one_thread()` first: an inherited `OMP_NUM_THREADS=4` is refused, not silently kept.

### 2.1 `s2io.py`, `setio.py`

```python
# s2io.py
STUDY = Path(__file__).resolve().parents[2]
class S2Error(RuntimeError)
def require_one_thread() -> int            # each of the three variables must be exactly "1"; returns 1
@contextmanager
def r2()                                   # gait.set_accuracy("r2"); assert on entry and exit; restore the previous level
def canonical(obj) -> bytes                # json.dumps(plain(obj), sort_keys=True, separators=(",", ":"), allow_nan=False)
def sha256_json(obj) -> str;  def sha256_file(path) -> str
def write_json(path, obj) -> None          # indent=1; path + ".tmp"; flush, fsync, os.replace
def write_npz(path, arrays: dict, compressed=False) -> str    # np.savez(open(path + ".tmp", "wb"), **arrays); replace; sha256
def sweep_tmp(root) -> list[str]           # remove stale *.tmp below root (call it holding the lock)
class Lock:                                # with Lock(dir / ".lock"): ...   O_CREAT | O_EXCL; {pid, host, started_at}
def portable(x)                            # panels._portable: paths under STUDY made relative
def built_by(stage2_modules: list[str], own_outputs: list[str] = ()) -> dict
    # {commit, dirty, dirty_paths (less paths under own_outputs), panel_sources_sha256, stage2_modules {path: sha256},
    #  versions (panels.library_versions()), blas_threads, accuracy (gait.ACCURACY), at}

# setio.py   (read-only; imports stage1.control, stage1.tune, s2io; never sets.py or moves.py)
ROOT = s2io.STUDY / "stage2" / "sets"
SPEEDS = (0.30, 0.35, 0.40, 0.45, 0.50);  BAR_SPEED = 0.40;  RAMP_S = 0.375
TURN_RATES = {"0.35": 0.35, "0.5": 0.5, "pi4": math.pi / 4, "1": 1.0};  QUICK = "pi4"
def walk_name(v: float | None) -> str              # "nominal" or f"v{v:g}"
def change_name(a: str, b: str) -> str             # f"{a}_to_{b}"
def stop_name(steps: int, walk: str) -> str        # "stop2_v0.4", "stop1_v0.4"
def turn_name(rate: str, direction: str) -> str    # "turn90_pi4_left"
def parse_name(name: str) -> dict                  # the inverse, exact; raises on a name it did not make
WALK_SUBSET = ("file", "sha256", "frames", "dt_s", "period_s", "speed_m_s", "stride_m", "yaw_rate_rad_s", "strides",
               "window_s", "crouch_frac")
MOVE_SUBSET = ("file", "sha256", "frames", "duration_s", "kind", ("gaits", "a"), ("gaits", "b"), ("move", "causal"),
               ("move", "command_phase"), ("move", "kind"), ("turn", "angle_rad"), ("turn", "rate_rad_s"),
               ("turn", "direction"), "a1_over")
def entry_subset(kind: str, entry: dict) -> dict;   def entry_sha256(kind: str, entry: dict) -> str
def set_dir(set_name: str, body: str, root=ROOT) -> Path
def load(set_dir) -> dict                          # {"set.json", "refs.json", "transitions.json", "screen.json" | None}
def frozen(set_doc) -> dict | None                 # the latest freeze {index, manifest_sha256, entries {...}} or None
def read_inputs(set_dir, files, require_frozen=True) -> tuple[dict, dict]
    # checks every file's sha256 against its entry and the latest manifest; returns
    # ({"dir", "files": {f: {"sha256", "entry_sha256"}}, "freeze": {"index", "manifest_sha256"}},
    #  {"refs.json": {"references": ...}, "transitions.json": {"transitions": ...}})     (tune.read_inputs' second shape)
def stage1_view(run_dir, body) -> tuple[dict, dict]    # the pseudo-set "S1": the same two values for a Stage 1 run dir
def commands(plan, tr_doc) -> list[dict]           # the causal moves of a plan: index, name, kind, command_phase, ...
def add_events(track, plan, tr_doc) -> list[dict]  # appends one event per command to track.events; returns them
def build_track(p, set_dir, plan, omega, tr_doc) -> control.Track      # control.build_track + add_events
def min_planned_swing_s(track) -> float            # shortest swing run of the track's segments, reference seconds
```

### 2.2 `moves.py`

```python
@dataclass(frozen=True)
class GaitX:
    """A periodic gait as a move sees it: its GaitSpec on a body, or the stand."""
    name: str                      # the reference's name in the set ("v0.4"), or "stand"
    spec: gait.GaitSpec | None     # None: the stand
    H: float = math.nan
    yaw_rate: float = 0.0          # rad/s, for a periodic turn gait (Stage 1's six; look turns later)
    in_place: bool = False         # a turn in place: speed 0.0 (Stage 1's in_place_0.50)
    split: str = "statics"         # "exact" for a turn gait
    @classmethod
    def of(cls, name, spec, H, yaw_rate=0.0, in_place=False) -> "GaitX"
    @classmethod
    def stand(cls) -> "GaitX"
    def legacy(self) -> transitions.Gait
        """transitions.gait_of's constructor call, argument for argument: Gait(name, info["period"], duty,
        spec.lift_frac * H, 0.0 if in_place else info["speed"], yaw_rate, spec.head_amp, spec.tail_amp,
        spec.crouch * H, spec.zmp_iters, spec.zmp_gain, split), info = spec.describe(H); duty = spec.duty when
        spec.swing_time is None (Stage 1's value and type), else info["duty"] (F3)."""

@dataclass(frozen=True)
class Schedule:
    """Everything transitions.plan hard-codes, as data."""
    kind: str                      # "start" | "stop" | "walk" (Stage 1's three) | "turn"
    window: tuple[float, float]    # output [first, last) in gait phase; last is an integer
    fn: dict                       # phase -> T, duty, lift, v, w, amp, head_amp, tail_amp, crouch (as _param_fns)
    ramps: dict                    # recorded as given
    lift_min: float | None = None  # swings exist only for lift-offs >= this (a start: 0.0)
    lift_max: float | None = None  # ... and <= this (a stop: 0.0)
    command: float | None = None   # command phase, a multiple of 0.5; None: Stage 1's non-causal construction
    rule_f: bool = False           # committed footprints (section 3.1 item 4); needs command
    pre: float = 1.0               # computed strides before `first` (discarded warm-up)
    post: float = 0.25

@dataclass(frozen=True)
class Move:
    name: str                      # file stem
    a: GaitX
    b: GaitX
    schedule: Schedule
    builder: str                   # "start" | "change" | "stop_one_step" | "stop_two_step" | "turn" | "stage1"
    args: dict                     # the builder's arguments as JSON-plain values: with a, b this IS the move's identity
    causal: bool = False
    laws: "Laws" = TODAY           # section 3.9
    rules: tuple[str, ...] = ()    # ids of the pre-registered rules it was made by (recorded)
    def definition(self) -> dict   # {builder, args, specs {a, b}, causal, laws: laws.name, rules}; no function object

def legacy_schedule(a: transitions.Gait, b: transitions.Gait) -> Schedule
    """Stage 1's schedule: fn, ramps = transitions._param_fns(kind, a, b) (the same function objects), window =
    transitions.WINDOWS[kind], lift_min 0.0 for a start, lift_max 0.0 for a stop, command None."""
def start(name, b: GaitX) -> Move
def change(name, a: GaitX, b: GaitX, ramp_strides: float = 1.0) -> Move     # refuses a.crouch != b.crouch
def stop_one_step(name, a: GaitX) -> Move
def stop_two_step(name, a: GaitX) -> Move                                    # D8; command -0.5, not an argument
def turn(name, a: GaitX, angle: float, rate: float, stride_factor: float, ramp_s: float = setio.RAMP_S) -> Move   # D6
def stage1(body: str, name: str, H: float) -> Move      # one of transitions.TRANSITIONS on a Stage 1 body
def stage1_gait(body: str, name: str, H: float) -> GaitX

def plan(move: Move, geo: transitions.Geometry, n: int = 160) -> transitions.Plan
def generate(src: dict, move: Move, ref_a: refs.Reference | None, verbose: bool = False) -> dict
    """transitions.generate's dict (same keys) plus: move, command_phase, causal, zmp_consistency_history_m,
    causal_transient_m, post_window_zmp_error_m, kappa (turns), ik_fail_frames."""
def kinematic(src: dict, move: Move, ref_a) -> dict     # one solve, no ZMP correction: q, ik_ok, ik_fail_frames, range_use
def arrays(g: dict) -> dict        # transitions.arrays(g) exactly for TODAY's laws; look laws append keys after it
def summary(g: dict, ref_a, ref_b, q_home) -> dict      # section 3.6; no wall time
def pick_stride_factor(src, a: GaitX, ref_a, angle, rate, grid=STRIDE_GRID, limit=TWIST_KINEMATIC) -> dict   # D7
def legacy_check(bodies, names=None, procs=1, record=None) -> dict          # D4's record
```

`moves` writes no set file: `generate`, `arrays` and `summary` return values, and `sets` places them with `s2io`.
A `Move` holds lambdas (`Schedule.fn`, `Laws`) and is neither picklable nor hashable: workers get `(set, body, item
name)` and rebuild it; records hold `move.definition()`.

CLI: `python -m designlab.stage2.moves legacy --body B [--name N ...] [--procs 1] [--stage1-run stage1/runs/2026-11-08]
[--record stage2/sets/legacy_check.json]` regenerates Stage 1 transitions through `moves` into a temporary
directory, compares each npz's sha256 with the run's `transitions.json`, writes or merges the record when
`--record` is given (per body; only a run of all 18 names sets `complete: true`), and exits 1 on a difference.

### 2.3 `sets.py`, `setscreen.py`, `readings.py`, `bar.py`, `regress.py`

```python
# sets.py
@dataclass(frozen=True)
class SetDef: name: str; role: str; spec_of: Callable[[str, float, float | None], gait.GaitSpec]; label_of: Callable; note: str
SETS: dict[str, SetDef]                                   # "V0", "F25F9", "F25F9cl" (V6, V6noF5 added by the look phase)
def v0_spec(body, H, v=None) -> gait.GaitSpec             # D3; v None: the nominal walk
def tuned_f9_spec(body, H, v) -> gait.GaitSpec            # the PR's function, verbatim
def items(set_name, body) -> list[Item]                   # Item(name, kind, depends, definition): walks, then moves
def walk_arrays(src, spec) -> tuple[dict, dict, dict]     # (Stage 1-format arrays, look-format arrays, entry parts), in s2io.r2()
def build_item(set_name, body, name, root=setio.ROOT) -> dict      # one item in this process (workers call it)
def build(set_name, body, root=setio.ROOT, only=None, procs=2, log=print) -> dict   # resumable; under the set's lock
def verify(set_name, body, root=setio.ROOT) -> dict       # section 4.3
def freeze(set_name, body, root=setio.ROOT) -> dict       # D20
def status(set_name, body, root=setio.ROOT) -> dict       # items done / missing / not available, frozen or not
# CLI: python -m designlab.stage2.sets {build|verify|freeze|status} --set S --body B [--only NAME ...] [--procs 2] [--root DIR]

# setscreen.py
def screen_walk(src, set_dir, name, look_entry) -> dict   # screen.screen_rung's block + range margins, no wall time
def screen_move(src, set_dir, name, entry, walk_block) -> dict   # section 3.7
def run(set_name, body, root=setio.ROOT, procs=2) -> dict # resumable per item; merges into screen.json
def markdown(doc) -> str
# CLI: python -m designlab.stage2.setscreen {run|md} --set S --body B [--procs 2]

# readings.py (no MuJoCo, no Stage 1 import; numpy and scipy.signal only; every function takes plain arrays)
def lowpass(x, fs, hz=1.0, order=2)                       # zero-phase Butterworth, equal to stage1.metrics.lowpass (tested)
def air_runs(contact_i, dt) -> list[tuple[int, int]]      # complete (lift-off step a, touchdown step b) runs of one foot
def toe_clearance_own_swing(t_u, low, contact, dt, ref_swing, ta, tb, air_gap_max_s, mid=(0.25, 0.75)) -> dict
def line_reading(t_u, com_xy, heading, ta, tb, c_episode_start=None) -> dict
def turn_reading(t_u, yaw_unwrapped, t_c, psi_cmd, sign, angle, at_s, hold_s=3.0, tol_deg=10.0) -> dict
def touchdowns(contact, dt) -> list[tuple[float, int]]    # (time s, foot) of every touchdown
def stop_reading(t_u, com_xy, v_xy, contact, dt, t_c, heading, command_speed, stride_s) -> dict
def drive_ratios(force_sq, sustained, actuators, leg_mask, k0, k1) -> dict
def reference_heading_at(arrays: dict, tau_s: float, rate: float) -> float            # D9, G9
def reference_readings(arrays: dict, kind: str, walk: dict | None = None) -> dict     # what a reference's own file can say

# bar.py
@dataclass(frozen=True)
class BarCondition(panels.Condition):          # judge in ("bar_walk", "bar_turn", "bar_nudge", "bar_stop")
    set: str = ""; walk: str = "v0.4"; move: str | None = None; frac: float = 0.25; block: str = "seeds3042"
def conditions(set_docs: dict) -> list[BarCondition]    # the 16 gated and the reported ones the set has references for
class BarPlanner:   resolve(cond) -> dict(plan, seconds, command_ref_s, steady_ref_s, window, push, files, at_s)
class BarContext(panels.Context):  built(cond) -> BarBuilt | panels.Built   # stock conditions go to panels.Context.built
class BarBuilt(panels.Built):      legs, recorder(), push_onset(seed), push_force(), command, required
class Delayed(panels.Row):         # wraps any panels.Row: name, inner, k; describe(), sources(), spec(), from_spec(), episode()
def rows(body, names, stage1_run) -> dict[str, panels.Row]     # "R2/42@4ms", "R0@10ms", "<registered row>@<ms>ms"
def judge(b, out, rec, seed, lg=None) -> dict
def run_episode(row, ctx, cond, seed, relax=(), trace_root=None) -> dict
def failed_episode(cond, seed, reason, relax=()) -> dict;   def not_run_record(...) -> dict
def record_key(...) -> dict;  def condition_record(...) -> dict;  def counts(episodes, seeds, which="success") -> dict
def failure(...) -> dict;  def summarize(out, set_name, body, block) -> dict
class BarWorker;  def _worker(...)                      # top-level factory for tune.Pool
def evaluate(set_name, body, out, rows, conditions=None, block="seeds3042", procs=2, attribute="judged",
             traces=True, episodes=20, task=None, supersede=False, log=print) -> dict
def dry_run(...) -> dict                                # episodes, planned seconds, cost from measured rates; writes nothing
def reference_recorder(p, track, seconds) -> metrics.Recorder     # any track played kinematically (G9; panels' takes walks only)
def refcheck(set_name, body) -> dict                    # each reference through reference_recorder and the bar's readings
def reread(stage1_run, body, rows, conditions, out, records=None, row_modules=()) -> dict   # stock episodes re-run with
                                                        # the contact table; traces checked against `records` (G9, section 4.4)
def bar_sources() -> dict                               # sha256 of bar.py, readings.py, setio.py, speedfit.py
# CLI: python -m designlab.stage2.bar {run|dry-run|summary|refcheck|reread} --set S --body B --rows ... [--conditions ...]
#      [--block seeds3042|seeds5042] [--passed-of RUN_BLOCK] [--procs 2] [--attribute judged|all|none] [--no-traces]
#      [--episodes 20] [--task 3|4] [--supersede] [--stage1-run DIR] [--records DIR] [--row-module M] --out DIR

# regress.py
CASES: tuple                                             # section 4.2's list: (run, body, row module, rows, conditions, seeds)
def run(tier="fast"|"full", out_record=None, procs=1) -> dict      # stock eval into a temporary --out; compares; writes the record
def check(record) -> list[str]                           # the record against today's source hashes
# CLI: python -m designlab.stage2.regress {run|check} [--tier fast|full] [--record stage2/regress/regress.json] [--procs 1]
```

### 2.4 Edits to existing files

| File | Change | Why the default is bit-identical | Owner, when |
|---|---|---|---|
| `designlab/stage1/control.py:967-978, 2300` | `Track.add_periodic`: `K = int(ref.arrays["window_strides"]) if "window_strides" in ref.arrays else 1`; `frame_info(..., period=ref.period / K, frames_per_stride=ref.n // K)`; `Segment` gains `stride_s: float = 0.0`, set to `ref.period / K` for a periodic segment and read at `:2300` in place of `sg.period` | no Stage 1 or round-A file holds the key; `x / 1 == x` and `n // 1 == n` exactly; `seg.period` (the window) keeps driving the wrap at `:1029-1032` | WP G, before state A, with section 4.2's full regression |
| `designlab/stage2/screen.py:516` | `SPEED_LABEL[speed]` -> `SPEED_LABEL.get(speed, speed)` in the "over" sentence | every committed speed key is in the dict; `screen.py` is hashed by no record; the sentence is only reached for a rung over a limit | WP C |
| `tools/restore_inputs.py`, `tools/README.md`, `tests/test_restore_inputs.py` | a `sets` group: its listing is made from the frozen manifests (`stage2/sets/*/*/set.json`), `NEEDS = {"sets": ("picks-models", "picks-arrays")}`; restoring calls `sets build` (which on a frozen set only places arrays that match); the test's pinned group tuple (`:96`), file count (`:79`) and the README's counts change in the same commit | new group only; existing groups' listings untouched | WP S, due at the freeze |

Not edited: `gait.py`, `turning.py`, `sizing.py`, `envelope.py`, every other `designlab/stage1/*.py`, `look.py`,
`motion.py`, `clips.py`, every committed record. By ruling 8 the Jetson body's merge will edit
`designlab/build.py`, `mjcf.py`, `stage1/__init__.py` and `stage1/refs.py` before the freeze; nothing here assumes
those files stay byte-identical, and every hash this round pins is read at run time, never written as a constant.

## 3. Algorithms

Notation: gait phase `p` in strides; left lift-offs at integers, right at half-integers (`transitions.py:325`);
`n = 160` frames per stride; `s(x)` the minimum-jerk step `transitions._ramp`; `T, d, v, w` period, duty, path speed
and yaw rate; a move's output window is `[first, last)`.

### 3.1 `moves.plan(move, geo, n=160)`: `transitions.plan` with the schedule as data

The body is `transitions.plan`'s (`transitions.py:296-426`), line for line, with these substitutions and no other
change (prototype `$D/proto_moves.py: gplan`, E1):

1. `fn, ramps = schedule.fn, schedule.ramps`; `first, last = schedule.window`; `lo, hi = first - schedule.pre,
   last + schedule.post`. `first` may be any multiple of `1/n`; `last` is an integer.
2. Swings: `events = [e for e in events if e >= lift_min]` when `lift_min` is not None, `<= lift_max` likewise
   (Stage 1: `kind == "start"` and `kind == "stop"`, `:327-330`).
3. `from_stand = move.a.spec is None`, `to_stand = move.b.spec is None` replace `kind == "start"` / `"stop"`.
4. **Rule F (committed footprints), when `schedule.rule_f`:** a stance whose `stance_from < command - PHASE_TOL`
   takes its footprint from `pose_pre(mid-stance)`, the path integrated with gait a's own constant `T, v, w` from
   the same origin, instead of `pose(mid-stance)`. A foot on the ground when the command comes keeps the footprint
   it had. The rule is complete only when the command is a lift-off (then no swing is in the air at the command),
   so `plan` raises `ValueError` for a causal schedule whose `command` is not a multiple of 0.5 within `PHASE_TOL`.
   The pre-command reference of every causal move is built from gait a's own constants; `rule_f` only switches the
   footprints (the stop needs no switch: its footprints are the walk's up to the last step).
5. The laws of section 3.9 replace `:378` (lift), `:385-387` (ZMP target), `:420` (pelvis height), `:421-422`
   (neck, tail) and add foot pitch and trunk roll. `TODAY` is those lines verbatim.
6. The returned `transitions.Plan` has `kind = schedule.kind if schedule.kind in WINDOWS else "walk"`, so Stage 1's
   `com_path` and `compare` read it unchanged.

Legacy exactness: `legacy_schedule` passes the function objects `_param_fns` returns and `WINDOWS[kind]`; items 4-5
are off; the arithmetic is then Stage 1's own.

### 3.2 `moves.generate`: two solve paths

**Non-causal, today's laws** (Stage 1's 18, `start`, `change`, `stop_one_step`): exactly `transitions.generate`'s
body (`transitions.py:638-678`) with `pl = moves.plan(move, geo)` and the gaits from `GaitX.legacy()`; it calls
`transitions.initial_state`, `com_path`, `solve_frames`, `derivatives`, `multibody_zmp`, `idyn` themselves.
`walk = b if a.stand else a`; `geo = Geometry.of(M, q_home, walk.crouch)`; passes and gain from `b` (from `walk` for
a stop), as `:648-649`. It begins with `gait.set_accuracy(refs.ACCURACY)` as Stage 1 does; callers wrap it in
`s2io.r2()` all the same.

**Causal** (`move.causal`; prototype `$D/proto_causal.py`, E4, E6, V1). `c = schedule.command`; the plan is made with
`first = c`, `pre = 1.0` (one warm-up stride that reproduces the walk and is discarded), `last` as scheduled.

1. *The walk's correction.* `delta[k] = com_ref[k] - com_ref''[k] / omega^2 - zmp_ref[k]` over `ref_a`'s 160 frames,
   the second derivative taken spectrally on the periodic part (`com_ref` less `k/n` of the stride advance). This is
   what `gait.generate` added to the ZMP target, recovered from the saved LIPM path ([proto] up to 39 mm forward,
   29 mm lateral). Requires `spec.sway_gain == 1` (refused otherwise).
2. *References.* `z_pre` = the ZMP target of the continuing walk's plan (the same schedule with gait a's constants,
   window to `c + ceil(2.5 s / T_a) + 2`) plus `delta` by frame. `z_0 = plan.zmp + amp(p) x R(heading(p))
   delta[frame(p)]` is the first reference of the commanded plan; before the command the two are equal.
3. *State.* `x0` = the walk's `com_ref` state at phase `c - 1` by `initial_state`'s fourth-order differences, with
   the frame offset generalised to any multiple of `1/n` and the walk shifted so that the plan's path origin is at
   phase `first`.
4. *Preview control with new information.* Same gains (`transitions.preview_gains`), same uniform grid at the
   smallest frame interval, same cubic read-out. Grid steps before the command frame preview `z_pre`; steps from it
   on preview the commanded reference. The control law runs in its **incremental form** after the first step,
   `u_k = u_{k-1} - Gi e_k - Gx (x_k - x_{k-1}) - Gp . (z[k+1 : k+1+N] - z[k : k+N])`, `z` the sequence in force at
   step k; the first step is `transitions.preview_run`'s position form. (The position form across a change of
   previewed sequence gives a jerk impulse, 430-560 mm of ZMP error at the command frame in the prototype; for one
   sequence the incremental form is the same controller: [review] the warm-up stride reproduces the walk to
   0.012 mm.) The function also returns the cart-table ZMP of its own cubic, `z_ct = c - c'' / omega^2`.
5. *Correction, with the excused transient bounded to one stride.* `fade(p) = 1 - s((p - c - 0.5) / 0.5)`: 1 up to
   half a stride after the command, 0 from one stride after it (`CAUSAL_FADE = (0.5, 1.0)`, recorded in the move's
   rules). For `it in range(iters + 1)`: CoM path (4); `solve_frames`; `derivatives`; `multibody_zmp`;
   `tau_c = fade(p) x (z_ct - zref)` on frames from the command on, 0 before (what a controller without
   foreknowledge cannot track in its first stride); `e = nan_to_num(zmb - plan.zmp - tau_c)`, `e[: k_cmd] = 0`;
   append `max |e|` over the output frames to `zmp_consistency_history_m`; `zref -= gain * e` unless last. From one
   stride after the command the loop therefore corrects `zmb - plan.zmp`, as Stage 1's does. Recorded per pass:
   `max |zmb - plan.zmp|` over all output frames (`zmp_error_history_m`, Stage 1's meaning), `max |tau_c|`
   (`causal_transient_m`), and `max |zmb - plan.zmp|` over the output frames with `p >= c + 1`
   (`post_window_zmp_error_m`). `iters`, `gain`: gait a's.
6. *Output.* Frames `[k_cmd, last)`; `t` starts at 0 at the command frame; inverse dynamics on the output frames with
   `idyn(..., split)`, `split = "exact"` for a turn, else the gaits'.

Cost [proto]: 38 s for the stop (760 computed frames), 77 s for the quick turn (1160), one process, two other cores
busy.

### 3.3 Joins and gates of a move (G3's kind)

- `join_b` (every move): `transitions.compare(g, ref_b, "b", q_home)`: the last output stride against gait b's
  periodic stride carried to the final path pose (or the home stance): CoM within 1.0 mm, multibody ZMP within
  2.0 mm.
- `join_a`: non-causal moves only, `transitions.compare(g, ref_a, "a", q_home)` (recorded; gated at the same 1.0 /
  2.0 mm; Stage 1 does not gate it and its own value is recorded as it was). **For a causal move `join_a` is
  `null`**: its output starts at the command and departs from the walk at once, so Stage 1's first-stride
  comparison has no meaning there.
- `join_command`, causal: the warm-up stride and the command frame against the walk's own frames: model CoM within
  1.0 mm over the warm-up stride and at the command frame; joints within 1e-3 rad at the command frame; multibody
  ZMP within 2.0 mm over the warm-up frames up to two frames before the command. The ZMP at the command frame
  itself is recorded (`at_command.zmp_mm`, with the note that its central difference straddles the command:
  [proto] 1.99 mm) and not gated.
- Limits on the ZMP histories: non-causal `zmp_error_history_m[-1] <= 4.0e-3` (Stage 1's largest recorded value is
  3.8 mm); causal `zmp_consistency_history_m[-1] <= 2.5e-3` and `post_window_zmp_error_m[-1] <= 4.0e-3`.
- Causal moves: single-support CoP margin `>= +0.010 m` (twice `control.COP_INSET`; the tracker's CoP feedback
  works inside the pad inset), recorded with its frame and foot ([proto] 24 mm in the turn). Stage 0's -10 mm rule
  (`look.COP_RULE_M`) stays what it is in the screen: the flag for "drive loads meaningful".
- `pass_ = ik_ok and join_b ok and (join_a ok or join_command ok) and the history limits and (causal: the CoP
  margin)`. A move that fails is retried by D21 or D7 and otherwise written with `pass_: false` and not frozen.

The gate constants above (1.0, 2.0, 2.5, 4.0 mm; 10 mm; the two-frame exclusion) were set on the 14 Nov prototypes
of V0 on `75-M-H-10`, before any closed-loop run; the rules text of every `transitions.json` says so.

### 3.4 The quick turn (`moves.turn`)

Inputs: walk a (the set's `v0.4`), signed rate `w0` (+ left), angle `A = pi/2`, stride factor `f`, ramp
`ramp_s = 0.375 s`. `T_w, d, lift, v` the walk's; `T_t = f T_w`; `r = 2 ramp_s / (T_w + T_t)` strides (a ramp's
duration is `integral T dp = r (T_w + T_t) / 2`).

- Command phase `p0`: the inside foot's lift-off: 0.0 for a left turn, 0.5 for a right turn.
- Shape `sigma(p) = s((p - p0) / r) - s((p - p1) / r)`; `w(p) = kappa w0 sigma(p)`; `T(p) = T_w + (T_t - T_w)
  sigma(p)`; `duty, lift, v, crouch` constant; `amp = 1`; **`tail_amp(p) = tail_amp x (T(p) / T_w)^2` and
  `head_amp(p) = head_amp x (T(p) / T_w)^2`** (ruling 1; rule id `S2A-appendage`).
- `p1 = p0 + L`, `L = (A / |w0| + (T_t - T_w) r (1 - 2 A2)) / T_t`, `A2 = integral_0^1 s(x)^2 dx = 0.391775`
  (closed form of `integral w T dp = A` for `L >= r`; refused otherwise). `kappa = A / |plan's own integrated
  heading|` with `kappa = 1` first, then one re-plan; `|kappa - 1| < 1e-6` required and recorded ([review]
  90.000000 deg).
- `rule_f = True`; `window = (p0, ceil(p1 + r) + 2)`; `causal = True`; `split = "exact"`. [review] the join is
  insensitive to 1.1 against 2.1 strides of lead-out.
- Gates beyond 3.3, read on the reference by `reference_readings`: heading total within 1e-9 rad of +-A; heading at
  bar time `at_s = A / |w0|` at least 80 deg on the nominal clock and on the slowest clock (D9); hip-yaw use at
  most 0.87 on the built reference.
- `pick_stride_factor` (D7): for f in the grid, descending: `kinematic()` of the left and the right turn at that
  f; stop at the first f whose two passes both have `ik_ok` and `max(range_use[*_hip_yaw]) <= 0.85`. `range_use`
  is `turning.summarize`'s formula (`turning.py:904-911`) on `plant.usable_ranges(body)`. Returned and recorded:
  every f tried with its two readings, the chosen f or `None`, the steady turn's kinematic reading at the chosen f
  (`turning.generate`, `zmp_iters=0`), the rule's text and constants, `steps_down`.

### 3.5 The two-touchdown stop (`moves.stop_two_step`)

Schedule = `legacy_schedule(a, stand)` (Stage 1's stop plan: speed and yaw-rate ramp over `[phi_c - 0.2, phi_c +
0.2]`, `phi_c = 0.5 - d/2`; amplitude ramp over `[0, 1 - d + 0.25]`; `lift_max = 0.0`) with `window = (-0.5, 3)`,
`command = -0.5`, `rule_f = False`, `causal = True`. Events, in phase: command -0.5 (right lift-off); right
touchdown `0.5 - d`; left lift-off 0; left touchdown `1 - d`; ZMP at the final stand from `1 - d + 0.25`. The track
plays `v0.4` for `N + 0.5` strides before it. Gates beyond 3.3, read on the reference: exactly two touchdowns
after the command; forward CoM speed below 5 % of the walk's speed within 1.5 strides of the command ([proto]
1.20); `max |1 Hz low-passed forward acceleration of the model CoM| <= 1.5 m/s^2` ([proto] 0.65). The rule text
states what this last gate can and cannot see: a zero-phase order-2 Butterworth at 1 Hz turns any speed change
`dv` into at most `2.22 x dv` m/s^2, so a stop from 0.42 m/s reads at most 0.93 m/s^2 however abrupt; the term
detects rocking, not abruptness, and the unfiltered CoM deceleration peak is recorded beside it ([proto] 2.16).
The command comes `(d - 0.5)` of a stride after the left foot's scheduled touchdown (V0 63 ms of reference time,
F25F9 47 ms, F25F9cl 85 ms): a left foot that lands that late lands after the command and counts.

### 3.6 Record schemas

Wall times (`seconds`) are in no committed JSON: they go to `logs/`. A rebuild that reproduces every array
rewrites nothing.

`refs.json`: Stage 1's top level (`body, accuracy, model, H_m, q_home, joints, feedforward, references`) plus `set`,
`schema: 1`, `rules`. Each entry: Stage 1's keys (`name, file, sha256, frames, dt_s, period_s, speed_m_s, stride_m,
lift_m, cadence_steps_s, yaw_rate_rad_s, stride_motion, min_cop_margin_single_m, max_friction_ratio, ik_ok,
generator, dynamics, spec, info` with `zmp_error_history_m`) plus:

```
set, label (null | "new"), margin_rule, crouch_frac, strides (1), window_s (= period_s when strides is 1)
look_file, look_sha256                     the look-format twin (screen, clips)
stage1_sha256, equals_stage1               where a Stage 1 file of the same spec exists (V0 nominal)
ladder_sha256, equals_ladder               where a look-ladder V0 rung of the same spec exists (V0 nominal, v0.4)
definition    {kind: "walk", spec (38 fields), body_model_sha256, generator: "look.generate + gait.inverse_dynamics + refs.arrays"}
built_by      s2io.built_by(...)           provenance, per entry
```

No `record_checks` (there is no round-two record to compare a new walk with; `refs.py:256-279` would raise).

`transitions.json`: Stage 1's top level (`body, transitions, method, preview, shift_strides, settle_strides,
path_half_width, walk_ramp_strides, windows, g3_com_mm, g3_zmp_mm`) plus `set`, `schema`, `rules` (every rule text
and constant of sections 3.2-3.5 used, with the note on where the constants were set). Each entry:
`transitions.summary`'s keys (`name, kind, from_, to, gaits {a, b}, output_phase, frames, duration_s, ramps, steps,
zmp_iters, zmp_gain, zmp_error_history_m, split, ik_ok, com_tracking_max_m, min_cop_margin_single_m,
min_cop_margin_m, max_friction_ratio, join_a, join_b, g3, file, sha256`; `stop` is `null` for a causal move) plus:

```
move            {kind, builder, causal, command_phase, command_t_s (0.0 when causal), rules [..], laws}
definition      move.definition() + {body_model_sha256, ref_a_sha256, ref_b_sha256}      the entry's identity
built_by        s2io.built_by(...)                                                        provenance
ik_fail_frames  int
range_use       {joint: largest fraction of the usable range reached}       hip_yaw_range_use: the worse leg's
join_command    causal only: {warmup_com_max_mm, warmup_zmp_max_mm, at_command {com_mm, joint_rad, zmp_mm, zmp_note}, ok}
zmp_consistency_history_m, causal_transient_m, post_window_zmp_error_m [per pass]
cop_margin_single_min {m, frame, foot}
turn            turns only: {angle_rad, rate_rad_s, direction, first_swing, ramp_s, ramp_strides, hold_strides,
                 stride_factor, period_in_turn_s, kappa, heading_total_rad, appendage_rule, stride_rule {text, grid,
                 limit_kinematic, limit_final, tried [{f, left, right, ik_ok}], steady_turn_kinematic, steps_down}}
reference       {at_s, heading_at_bar_deg, heading_at_slowest_clock_deg, heading_at_ref_clock_deg, time_to_80deg_bar_s}  (turns)
                {touchdowns_after_command, touchdown_s [..], forward_stopped_s, forward_stopped_strides,
                 accel_forward_lp1hz_peak_m_s2, accel_forward_raw_peak_m_s2, margin_before_command_s,
                 stage1_stop {command_phase (Stage 1's, labelled), ...transitions.stop_time_arrays' dict}}       (stops)
                {accel_forward_lp1hz_peak_m_s2}                                                                 (every move)
a1_over         true when a neck or tail planning limit is exceeded at Stage 0 (set by setscreen; D7)
fallback        null | {rule: "D21", step: 1 | 2, tried [...]}
pass_           section 3.3
```

`gaits.b.duty` stays where `tune.build_spec` reads it (`tune.py:484`). An item with no reference is an entry too:
`{name, kind, available: false, reason, numbers}` with no file.

`set.json`: `{set, body, role, note, spec_rule, speeds, crouch_frac, omega_rule, rules, look_entries {walk: entry
of look.write's shape, less wall times}, stride_rules {rate: pick_stride_factor's record}, not_available {name:
{reason, numbers}}, freezes [{index, at, commit, manifest {name: {kind, sha256, entry_sha256}}, manifest_sha256,
legacy_check_sha256, mixed_sources, dirty}]}`.

`legacy_check.json`: `{rule, bodies {body: {complete, names {name: {sha256, recorded, equal}}, passes}},
moves_sha256, panel_sources_sha256, versions, blas_threads, commit, dirty, at}`.

### 3.7 Stage 0 screen of every reference (`setscreen`)

*Periodic walks.* `sets.build` writes each walk in two formats from one generation (D3): the Stage 1 reference and
the look ladder's (`look.arrays(src, g)`), with an entry of `look.write`'s shape (`file, sha256, frames, spec, info,
measured = look.measure(...)`, `price = look.price(...)`, `flags`, `features = []`, no wall times) under
`set.json`'s `look_entries`; nothing goes through `look.prepare` or near `stage2/look/`. `screen_walk` =
`screen.screen_rung(src, walk_name, set_name, look_entry, out_dir=<set dir>)` with two changes to its result: the
block's `seconds` is dropped (it goes to the log), and a `range` block is added from
`look_entry["price"]["clearance"]["joint_range_margins"]` (`screen_rung` removes that key, `screen.py:495`): every
joint's margins against the usable range, the tightest, and the 0.10 rad gate.

*Moves* (`screen_move`), from the transition npz (`q, qd, qdd, tau, omega, swing, t, com, feet, heading, path`):

| Block | How |
|---|---|
| `drives` | `sizing._motor_space(names, tau, omega, layout)` -> `envelope._ratios(design, bus, ms)` over the output frames: window RMS / sustained, 1.25 peak / planning peak, 1.15 speed / speed at the bus; all eight groups |
| `passive` | `screen.passive_loads(M, design, tau, omega)`; `over_with_passive` listed |
| `summary` | `min_cop_margin_single_m`, `max_friction_ratio`, `ik_ok`, `ik_fail_frames` from the entry |
| `friction` | per single-support frame, `turning.inverse_dynamics`' formulas (`turning.py:807-824`: `mu_req`, `mu_req_cop`, `util` at mu 0.8) from `turning.contact_torques(..., "exact")`; double support `turning.ds_min_mu`; `grips {0.9, 0.54}` |
| `sway` | lateral CoM excursion in the path frame, peak to peak per stride of the move, / H |
| `accel` | model CoM on a uniform 500 Hz resample of the move's own frames (no wrap): horizontal peak, forward (along the heading) 1 Hz low-passed peak, vertical range; feet vertical peak |
| `clearance` | `look._clearance_gates(src, dict(q=..., phases=..., ik_ok=...))`, the move standing in for the nominal walk, on the round-two MJCF, `substeps = 0` (no pose is interpolated across the window's ends); `phases` from the move's planned `swing`. `clearance._swing_runs` is cyclic, so `screen_move` first asserts that no foot is in swing on both the first and the last frame (true of every move built here: each window ends at a stride boundary in double support or in a stand) and refuses otherwise |
| `range` | per joint, margins against the usable ranges; `hip_yaw_range_use` |
| `vs_walk` | each headline number against the block of the move's own walk (gait a, or b for a start) |
| `verdict` | `within_limits`, `over [..]`, `near [..]`, `a1_over`, `blocks_freeze [..]` |

**Which gate blocks a freeze** (one table for walks, straight moves and turns; everything else is reported):

| Gate | Walk | Start, change, stop | Turn |
|---|---|---|---|
| a **leg** drive's cont, peak or speed ratio > 1 | blocks: the rung is dropped (plan 4 A.3) | blocks (D21) | blocks (D7: no reference) |
| a **neck or tail** drive's ratio > 1 (inverse dynamics, as Stage 0 reads it) | blocks (plan 4 A.3: goes to the owner) | blocks (D21) | does not block: `a1_over: true`, read under reading 3 (ruling 1) |
| `ik_ok`; single-support CoP (walks and non-causal moves: Stage 0's -10 mm; causal moves: +10 mm, section 3.3) | blocks | blocks | blocks |
| swing clearance: each planned swing's mid-swing minimum of the pads >= 0.04 H | blocks | blocks | blocks |
| interference along the frames; component overlap; tail clearance | blocks | blocks | blocks |
| joint range | 0.10 rad on every gated joint | 0.10 rad | hip yaw by D7 (use <= 0.87); every other joint 0.10 rad |
| forward 1 Hz acceleration <= 1.5 m/s^2 (decision 10) | reported | blocks for starts, changes and stops | reported |
| grip need at mu 0.9 and 0.54; sway against 0.12 H; passive neck and tail loads | reported | reported | reported |

`sets verify` reads these keys of a `screen.json` block and no others: `arrays.sha256`, `verdict.blocks_freeze`,
`verdict.a1_over`, `drives.<group>.{cont_ratio, peak_ratio, speed_ratio}`, `range.hip_yaw_range_use`. For the
owner's kinematic clips later: walks have the look-format file (`points`, `toe_edge`, `foot_quat`, ...); a move's
file has `q`, `t`, `feet`, `swing`, from which `look.markers` and the clip writer's other arrays are computed
without a rebuild.

### 3.8 The bar's readings, from the recorder

Arrays (`metrics.Recorder`, `metrics.py:103-206`): `rec.dt` (0.002 s), `rec.loads [steps, 2]`, `rec.weight`,
`rec.contact() [steps, 2]` (2 % of weight, 20 ms debounce), `rec.force_sq [steps, nu]`, `rec.sustained`,
`rec.actuators`, `rec.U [rows, 31]` at 100 Hz with columns `t, t_ref, com_x, com_y, v_x, v_y, low0, low1, roll,
pitch, yaw, acc_x, acc_y, ref_swing, ref_u`, `rec.Q`, `rec.poses`. By the recorder's own convention a row's `t` is
`d.time - dt`, "the start of the physics step just taken", and `loads[k]` is read after step k
(`metrics.py:138-178`), so the readings treat physics step `a` and a row with `t = a dt` as one instant. WP B
checks that convention once on a real episode (a foot's `low` at the row of its touchdown step is within one
step's travel of the floor) and records the result; if it is off by one step the readings shift `t_lo` and `t_td`
by that step, in one named constant.
`psi_cmd` = the commanded heading = `b.home_yaw` (every plan starts straight from the home stance); `h = (cos
psi_cmd, sin psi_cmd)`; `H = b.H`. `t_c = rec.sim_time(t_ref_cmd)`; if it is None the episode did not reach the
command: `fell_before_command` is true and the episode fails `survived`.

Every episode records `command_s` (`t_c`), `command_ref_s`, `fell_before_command`, `segment_reached`
(`control.run`'s), `window_reached` and `omega`, `crouch_frac`. An episode that did not fall and whose trace ends
before its judged window does is an **error** (run again, like Stage 1's failed episodes), not a failed term.

**The three readings.** `drive_ratios` gives each drive's RMS over a span of physics steps over its sustained
rating, the worst leg drive and the worst neck or tail drive (`plant.leg_drives(p.drives)` is the split: every
drive but the neck's, the tail's and Stage 2's neck yaw and jaw). The term `drive_sustained` is "every drive at or
under 1"; `drive_sustained_legs` is the same over the leg drives. `success` holds the item's terms with
`drive_sustained` (reading 2); `success_but_A1` holds them with `drive_sustained_legs` in its place (reading 3);
`succeeded` and `succeeded_but_A1` are their conjunctions. The span is the item's judged window for the walk and
the turn (Stage 1's `SUCCESS_RULE`) and the whole episode for nudges and the stop (Stage 1's `LIMITS_RULE`,
`panels.py:1084-1087`; the episode form must equal `control.run`'s `worst_sustained_ratio`, tested). Both spans'
per-drive ratios are stored in `measures.bar.drives`.

**(a) Toe clearance over each foot's own swing** (`toe_clearance_own_swing`) **[amends ruling 5's air-gap
clause]**. Per foot i:
- an *air run* is a maximal run of physics steps with `contact()[:, i]` false that starts after a contact step
  (lift-off step a, `t_lo = a dt`) and ends at a contact step (touchdown step b, `t_td = b dt`); `D = t_td - t_lo`;
- an air run is a **swing** if any update row with `t_lo <= t < t_td` has `ref_swing == i`, **whatever its
  length**, or if `D >= air_gap_max_s`. Otherwise (wholly inside the reference's stance and shorter than
  `air_gap_max_s`) it is an *air gap*: counted and reported (`air_gaps`, `air_gap_longest_s`), not read.
  `air_gap_max_s = 0.5 x min_planned_swing_s / RATE`: half the shortest planned swing of the condition's track,
  put on the simulated clock (ruling 5's threshold, now applied only to stance chatter). The ruling's plain rule
  dropped any air run under half a swing: a touchdown at 48 % of a 0.35 s swing, or a 30 ms stub at mid-swing,
  left the judged reading instead of failing it ([review]);
- a *reference swing* is a maximal run of rows with `ref_swing == i`; one that holds two or more swings of foot i
  is **interrupted**;
- a swing with `t_lo >= ta` and `t_td <= tb` is read as the minimum of `low_i` over: its value interpolated
  linearly between rows at `t_lo + 0.25 D` and at `t_lo + 0.75 D`, and every row strictly between those two times.
  (Rows alone over-read by up to 5.6 % of the lift as the lift-off crosses the 10 ms grid, V7; with the two edge
  values the reading does not depend on the grid.)
`toe_clearance_own_swing_m` = the smallest swing reading. Success: `>= 0.04 H`, no interrupted reference swing in
the window, and per foot `own_swings >= ref_swings - 1` (`ref_swings`: that foot's reference swings wholly inside
the window). Recorded: `own_swings`, `ref_swings`, `interrupted_swings`, `air_gaps`, `air_gap_longest_s`,
`air_gap_max_s`, and the per-swing table `(foot, t_lo, t_td, min, class)` in the episode's trace. Stage 1's
reading (`metrics.window`'s `toe_clearance_m`, on the reference's clock) is recorded beside it, unchanged.

**(b) Walk item** (`line_reading`), window `[ta, tb] = [t_end - 10.0, t_end]`, `t_end` the last update of a complete
episode (`panels.py:1109-1111`): `c0` = CoM at the first update at or after `ta`; `along_commanded_heading_m =
(c(tb) - c0) . h`; `lateral_from_line_max_m = max_t |(c(t) - c0) x h|` over the window's updates (ruling 5: the line
through the CoM at the start of the judged 10 s). Terms: `survived`; `along` (at least `10 s x` the rung's target
speed: 4.0 m at 0.40 m/s); `line` (at most 0.25 m); `duty` (0.55-0.65; 0.55-0.72 below 0.30 m/s, owner decision
9); `no_flight`; `roll_rms`, `pitch_rms`, `roll_peak`, `pitch_peak`; `toe_clearance_own_swing`; `slip_median`,
`slip_worst`; `drive_sustained` (or `_legs`); `drive_speed`; `joint_range` (0.10 rad). Reported, not judged:
`grf_peak_bw` (Stage 1 drops `grf`, `cadence` and `stride` for every walk but the nominal one,
`panels.NOMINAL_ONLY_TERMS`; the bar does the same for every item from a walk other than the nominal one),
`lateral_end_m`, `heading_drift_deg`, `lateral_from_episode_start_line_max_m`. **Stage 1's reading**, for the
comparison with JPS 2.3, is stored under `stage1`: `tune.measure(spec, rec, out, None, window)`'s success dict
restricted to `panels.terms_of`'s list for a non-nominal walk (`tune.SUCCESS["walk"]` less `NOMINAL_ONLY_TERMS`;
for `walk/nominal` the full list), with `succeeded` recomputed from that list. (`tune.measure`'s own `succeeded`
would be false in every episode at 0.40 m/s: cadence 2.44 steps/s against 1.6-2.2, stride 0.686 H against
0.5-0.65.)

**(c) Turn item** (`turn_reading`): `psi(t)` = `rec.yaw_unwrapped()` re-based so that `psi - psi_cmd` is the
unwrapped heading from the commanded one; `err(t) = psi(t) - psi_cmd - sign x 90 deg` (ruling 5: the start heading
is the commanded heading at the command). `at_s = (pi/2) / |rate|`. `heading_error_at_deg = err(t_c + at_s)`
(interpolated); `heading_error_hold_max_deg = max |err|` over the updates in `[t_c + at_s, t_c + at_s + 3.0]`;
term `heading`: every update in that interval within 10 deg. `time_to_within_10deg_s` = the first `tau >= 0` from
which `|err| <= 10 deg` holds to `t_c + at_s + 3.0`, None if never. Reported: `turned_own_deg = psi(t_c + at_s) -
psi(t_c)` and `trunk_heading_at_command_deg`. Limits over `W = [t_c, t_c + at_s + 3.0]` by `metrics.window(rec,
*W, psi_cmd, H)`: `duty, no_flight, roll_rms, pitch_rms, roll_peak, pitch_peak, slip_median, slip_worst`;
`toe_clearance_own_swing` by (a) on W; `drive_sustained` (or `_legs`) on W; `drive_speed`; `joint_range` by
decision 14, `required = max(min(0.10, ref_margins - 0.02), 1e-3)` with `ref_margins = spec.ref_margins` (the
whole track, `tune.py:514-520`; `build_spec` itself returns the walk's 0.10 for a custom walk plan, so the bar
computes this); `no_leg_interference` from `metrics.leg_gap(rec, ctx.lg)` with the recorder's `legs_every =
tune.LEGS_EVERY`. `grf_peak_bw` reported.

**(d) Nudges.** Pulse `(t0, 0.1 s, F h_dir)` through `control.run(push=)`: `F = m x frac x env.PUSH_BOUND_M_S[body]
/ 0.1`, `frac = 0.25` (Stage 1's panels use 0.5); direction in the commanded heading's frame
(`panels.PUSH_DIRECTIONS`); onset `t0 = (t_ref_steady + (2 + u) T_v0.4) / RATE`, `t_ref_steady` the track time at
which the `v0.4` segment starts, `u = panels.push_u(cond, seed)` uniform in [0, 1): a uniform phase of one stride.
The "command" of a nudge is its onset (`command_s = t0`). Reading: `panels.recovery(rec, t0 + 0.1, standing=False,
speed_cmd=0.40 x RATE, stride_s=T_v0.4 / RATE, limit_s=3.0, until_s=planned end)` (decision 30). Terms: `survived,
recovered, drive_sustained` (or `_legs`), `drive_speed, joint_range`.

**(e) Stop item** (`stop_reading`): `v_f(t) = (v_x, v_y) . h` at the updates; command speed `v_c = 0.40 x RATE`;
`t_stop` = the first update time `>= t_c` from which `|v_f| < 0.05 v_c` holds to the end of the trace (None:
never). A *touchdown* is a physics step a with `contact()[a, i]` and not `contact()[a - 1, i]`;
`touchdowns_after_command` = the number with `t_c < a dt <= t_stop` over both feet. Terms: `survived`;
`stopped_by_end` (`t_stop` is not None: the trace holds 2.0 s after the planned rest); `touchdowns_after_command`
(at most 2); `stop_acceleration_1hz` (`metrics.acceleration_peak(rec, t_c, t_c + 2 T_v0.4 / RATE)` at most
1.5 m/s^2; section 3.5 says what it can see); `drive_sustained` (or `_legs`), `drive_speed`, `joint_range`.
Reported: `forward_stopped_s`, `forward_stopped_strides`, `touchdown_table` (each counted touchdown's foot and
time), `last_touchdown_before_command` (foot, time, and its distance to the command in s: a late landing of the
earlier step is then visible), `touchdowns_to_end`, `deceleration_raw_peak_m_s2` (the CoM's, unfiltered), and
decision 18's reading from `control.run` (`out["stops"]`), which the stop's command event makes available.

Episode plans (`BarPlanner`), `L = [("transition","start"), ("ref","nominal",1), ("transition","nominal_to_v0.4")]`
(for a ladder rung the change to that rung, or D21's path through 0.40 m/s):

| judge | Plan | Simulated length |
|---|---|---|
| `bar_walk` | L + `("ref", walk, K)` | Stage 1's rule (`panels.py:634-642`, `WALK_RULE`): K the fewest strides with steady + 0.5 P, over 0.95, + 10 s x 1.05 of reference; `seconds = t_end / RATE` |
| `bar_turn` | L + `("ref","v0.4", 2 [+0.5 for a right turn])` + `("transition", turn)` + `("ref","v0.4", K)` | `seconds = t_ref_cmd / (0.95 RATE) + at_s + 3.0 + 0.2`; K the fewest strides with `t_end >= seconds x 1.05 RATE` |
| `bar_nudge` | L + `("ref","v0.4", K)` | `seconds = latest onset + 0.1 + 3.0 + 2.0` at the slowest clock; K as above |
| `bar_stop` | L + `("ref","v0.4", 2.5)` + `("transition","stop2_v0.4")` + `("hold", h)` | `seconds = t_ref_rest / (0.95 RATE) + 2.0 + 0.2`; h the hold that makes `t_end >= seconds x 1.05 RATE` |

A turn or stop episode is thus long enough at the slowest clock and its track long enough at the fastest. `spec =
tune.build_spec(p, set_dir, "walk", None, omega, refs_doc, tr_doc, RATE, plan=plan, gait=walk)` (a custom walk
plan may have no stop event; [review] confirmed), then `setio.add_events(spec.track, plan, tr_doc)`. Seeds:
`panels.episode_seeds(cond, seed)`: every bar judge takes slot 0 (D12).

**Record** (`<out>/<set>/<body>/<block>/eval/<row>/<behavior>/<condition>.json`), Stage 1's keys
(`panels.condition_record`) with: `schema: "bar-1"`, `set`, `block`, `status` (`done` | `not run`), `definition`
(`BarCondition.describe()`), `rules` (every rule text used: clocks, readings, the three readings, PASS, delay,
seeds, shared lead-in, omega, the margin label for F25F9cl, and Stage 1's `COMMON_RULES` and row rules it relies
on), `terms`, `counts` (reading 2), `counts_but_A1` (reading 3), `counts_look` (`null` + reason), `failure`,
`attribution`, `falls {before_command, after_command}`, `omega`, `crouch_frac`, `set_freeze {index,
manifest_sha256}`, `sessions [{started_at, ended_at, commit, dirty, versions, blas_threads, procs, seeds_run,
wall_s}]`, `provenance`, and

```
key = {schema, body, set, block, row: portable(row.describe()), row_sources: portable(row.sources()),
       condition: cond.describe(), plan, seconds, seeds, command: {name, ref_s, phase} | None,
       inputs: {file: {sha256, entry_sha256}},                # D14's named subsets; plus the row's gains file sha256
       context: panels.key_context(body), sources: {panels: panel_sources().sha256, bar: bar_sources().sha256}}
```

`bar.py` has its own `failed_episode`, `not_run_record`, `record_key`, `condition_record`, `counts`, `failure` and
`summarize` (the panels' call `terms_of`, `rules_of` and `files_of`, which raise on a bar judge). Episode record:
`panels._base`'s keys + `success, succeeded, success_but_A1, succeeded_but_A1, failed_terms, failed_terms_but_A1,
command_s, command_ref_s, fell_before_command, segment_reached, window_reached, measures {bar {the item's readings,
drives}, window, stage1, legs, push, recovery}`, `trace`. Trace
(`.../eval/traces/<row>/<behavior>/<condition>/<seed>.npz`): `panels.write_trace`'s five arrays unchanged plus
`contact` (bool `[steps, 2]`, bit-packed), `dt`, `weight`, `t100` and `low100` (float32 `[rows, 2]`: both feet's
low points at every update), `swings` (the per-swing table), `command_s`: enough to re-read (a), the line and the
touchdown count without a new episode. A "not run" record holds `reason` and the numbers of the missing
reference.

**No-loss and G9 on stock episodes.** For a `panels.Condition` (set `S1`), `BarContext.built` is
`panels.Context.built` and `judge` is `panels.judge`; the record has `definition.stock = true` and Stage 1's
terms. `bar reread` runs stock rows on stock conditions in this way and adds the contact table and (a) to each
episode; it first checks the stock trace's sha256 against the committed episode record, so the own-swing reading
is read on episodes proven identical to Stage 1b's (section 4.4).

### 3.9 The look in moves (R6)

`moves.plan` gets every law from `move.laws`. The signatures are fixed now; the look's implementations come later
in `moves_look.py`, which `moves.generate` loads by name when `laws.solver == "look"`, so the look phase edits
neither `moves.py` nor any test of the non-look phase.

```python
@dataclass(frozen=True)
class PlanCtx:                 # what a law may read; built by moves.plan before the laws are called
    phase: np.ndarray          # [N] computed frames, strides
    t: np.ndarray              # [N] s
    fn: dict                   # the schedule's phase functions (T, duty, lift, v, w, amp, head_amp, tail_amp, crouch)
    amp: np.ndarray            # [N] the amplitude ramp
    weight: np.ndarray         # [N] share of gait b in [0, 1] (0: gait a's law, 1: gait b's; the schedule's ramp)
    heading: np.ndarray        # [N] rad
    swing: np.ndarray          # [N, 2] bool;   swing_u: [N, 2]
    swings: list               # (side, lift_off, touchdown, lift, footprint from, footprint to)
    stances: list              # (side, from, to, footprint)
    geo: transitions.Geometry
    a: GaitX;  b: GaitX;  command: float | None

@dataclass(frozen=True)
class Laws:
    name: str                                                      # "today" | "look"
    lift: Callable[[np.ndarray, PlanCtx, tuple], np.ndarray]       # (u in [0,1], ctx, swing) -> lift fraction;  today: sin(pi u)^2
    height: Callable[[PlanCtx], np.ndarray]                        # -> pelvis z [N];    today: geo.root0_z - amp x crouch(phase)
    neck: Callable[[PlanCtx], np.ndarray]                          # -> rad [N];         today: amp x head_amp(p) x sin(4 pi p)
    tail: Callable[[PlanCtx], np.ndarray]                          # -> rad [N];         today: amp x tail_amp(p) x sin(2 pi p + pi/2)
    trunk_roll: Callable[[PlanCtx], np.ndarray] | None = None      # -> rad [N] about the heading axis; None: 0
    foot: Callable[[PlanCtx], dict] | None = None                  # -> {pitch [N,2], contact_offset [N,2,3], rolling [N,2]}; None: flat
    zmp_point: Callable[[PlanCtx, tuple], np.ndarray] | None = None   # (ctx, stance) -> world xy; None: the pad centroid
    period_factor: Callable[[PlanCtx], Callable] | None = None     # -> phase -> factor on fn["T"] (F14's step times); None: 1
    solver: str = "stage1"     # "stage1": transitions.solve_frames, idyn, initial_state, compare
                               # "look":   moves_look.solve_frames, idyn, initial_state, compare, arrays_extra
TODAY: Laws
```

What today's code hard-codes, and the rule that carries each feature through a move (each is a rule a reviewer can
check on the arrays):

| Feature | Hard-coded at | Rule in a move |
|---|---|---|
| F3 `swing_time` | `transitions.py:173` reads `spec.duty`; `turning.py:396` | L1: a gait's duty is `describe(H)["duty"]` (`GaitX.legacy`); inside a ramp the duty mixes with the ramp's weight like every other parameter (sampled at lift-off, `:331`) |
| F13 `stride_rel` | none (via `describe`) | free |
| F9 `tail_phase`, `tail_amp` | `:422` `+ pi/2` | L2: `tail = amp x tail_amp(p) x sin(2 pi p + tail_phase(p))`, amplitude and phase mixed with the ramp's weight; a stand has amplitude 0; in a turn's quicker steps `tail_amp(p)` and the neck's amplitude carry ruling 1's `(T(p)/T_walk)^2` |
| F10 `head_mode`, settle | `:421` nod | L3: stabilise: `neck = -head_gain x trunk pitch about home` (zero in these plans) plus the settle dip after each touchdown of the plan (`settle_lag`, `head_settle`), scaled by `amp` and the ramp's weight between a nodding and a stabilising gait |
| F4 `lift_peak` | `:378` | L4: `gait.lift_profile(u, lift_peak)`, `lift_peak` sampled at the swing's lift-off |
| F1/F2 `bob*` | `:420` | L5: pelvis `z = root0_z - amp x (crouch + bob(p) x gait.bob_profile(tau))`, `tau` the fraction of the interval between the plan's consecutive touchdowns, trough by `gait.bob_trough`; bob amplitude mixed with the ramp's weight; 12 correction passes (`look.LOOK_ZMP_ITERS`), from the gait's spec |
| F6 `roll_amp` | `:570` yaw only | L6: trunk quaternion = yaw(heading) x home x roll(`amp x roll_amp(p) x` the walk's roll law in phase); in a turn the hip-roll margin is screened (section 3.7) |
| F14 `strides, phase_jitter, asym, gaze_var, tail_var` | `:359` uniform phase; one stride per file | L7: step-time deviations enter the period step by step (`period_factor`, `gait._Warp`'s rate); L8: the variation envelope is 1 on the frames a move shares with a periodic window (its warm-up and join strides) and fades with `s` to 0 across the next stride inward; between the fades a move has no F14 variation |
| F5 `toe_off*`, `toe_reach` | `:568` flat feet; `:385` pad centroid; `:681` flat-pad split | L9: per stance, the toe-off pitch of `gait.toe_off_bump` about the yawed foot's own lateral axis, the contact offset `gait.roll_offset` rotated by the foot's yaw, amplitude `amp x toe_off(p)` sampled at the stance's lift-off; `gait.rolling_entry` evaluated per double support on the two actual footprints; inverse dynamics by `look.look_inverse_dynamics`' rolling split, frame by frame. **L10 (the pre-registered fallback), stated per stance so that it is causal:** in a commanded move every stance that ends after the command is flat (toe-off amplitude sampled at its lift-off is 0, no rolling entry at its start); the swing that lifts at the command carries the walk's toe-off pitch to zero by its touchdown; toe-off returns with the first stance that begins after the move's last ramp; a stop's two remaining stances are flat. The frames before and at the command equal the look walk's, foot pitch included (tested) |
| F7 `zmp_inset_y` (V6S only) | `:385` | along the yawed foot's lateral axis (`zmp_point`); not needed for V6 |
| F11, F12 | not fields | not built in round A |

(The draft's L10 faded the toe-off "over the stride before any ramp": that is before the command and would make
the look's commanded moves non-causal. In the V0 turn the outside foot's stance at the command ends half a stride
after it and V6's toe-off starts at 0.825 of stance, after the command, so the per-stance rule costs nothing.)

**A K-stride periodic reference.** File: Stage 1's keys with `n = 160 K` frames and `stride_motion` of the whole
window, then `window_strides` (K) and the look keys (`phase, swing_u, foot_pitch, rolling, toe_edge, trunk_quat,
...`); `tau, omega, cop_margin, normal, friction` from `look.look_inverse_dynamics`. Entry: `period_s` = the
stride period, `window_s`, `strides` = K. A plan item `("ref", name, x)` counts **windows** of the file
(`Reference.period` is the window): `BarPlanner` and every other planner take a periodic item's duration from
`window_s`, never from `strides x period_s`.
- *Joined (J1-J3).* F14 samples the window uniformly in time and warps the phase, so inside a window no lift-off
  falls on a frame or on a multiple of T (V6: up to 8.9 ms, 3.5 mm of CoM travel); only the window boundary is
  exact. J1: a move leaves a K-stride walk at the warp's own time of its command phase, `t_leave =
  _Warp.time(j)` or `_Warp.time(j + 0.5)` for a right-foot command; the track plays `m + t_leave / W` windows
  (`add_periodic` takes a float); the move's warm-up, `x0` and `join_command` frames are the window **resampled on
  the move's phase grid** (the window is C2 in time: `q`, `com_ref`, `zmp_mb` through `_Warp.phase`); the entry
  records `a_phase` and `a_time_s`. J2: a move enters a K-stride walk at its window boundary: its last output
  stride is compared (`join_b`) with the window's last stride resampled the same way, and the walk then starts at
  frame 0. J3: by L8 the move's own frames carry the window's variation on those shared strides only.
- *Consumed by the track.* With section 2.4's edit: `joint(t)` and `info(t)` wrap by the window (`seg.n`,
  `seg.period`, as today); the feedback's stride period (`info["period"]`: the heading buffer at
  `control.py:1773`, the step offsets' decay at `:1814`) and `_anchor`'s half stride (`:885`) are the stride's.
  The estimator reads no period. Other readers of `Reference.period` as one stride, to be changed with R5 (state
  B, outside `panel_sources`): `env.py:385, 478, 629` (the phase input, `stride_s`) and `panels.py:2173, 2233`
  (`reference_recorder`, `g7`; the bar has its own recorder).
- *Periodic look turns* (R5's walking turns at 0.35-1.0 rad/s and the turn in place): not from `turning.generate`,
  which refuses a look spec and solves a one-stride FFT. They are cut from the steady part of a long `moves` plan
  (enter, hold N strides, exit) and gated for periodicity (consecutive hold strides mapped by the stride motion:
  CoM 1.0 mm, ZMP 2.0 mm, joints 1e-3 rad). Same generator, same laws, no edit to `turning.py`.

**Cheap, hard, and what to cut:**

- Cheap (a day together): F3 (L1), F13, F9 (L2), F10 (L3), F4 (L4).
- Moderate: F1/F2 (L5; 12 passes double the build time), F6 (L6), the periodic look walk in Stage 1 format with
  feedforward (re-uses `look.look_inverse_dynamics`), look turns cut from long plans.
- Hard: F14 in moves (L7-L8, J1-J3 with the resampling); F5 in moves (L9: rolling contact under a yawing trunk,
  `rolling_entry` on arbitrary footprints, the rolling split for finite frames).
- Cuts with the least loss, in order: (1) F5 inside turns and stops: L10 instead of L9 (periodic walks keep F5;
  it is also what `V6noF5` needs, so nothing extra is built); (2) F5 inside starts and changes too; (3) J1's
  resampling: "a move leaves and enters a K-stride window only at its boundary" (a right-foot command at the
  exact time of phase 0.5 after the boundary); episodes then walk whole windows (about 10 s) before a command;
  (4) F14 inside periodic turns (one-stride turn references; F14 stays in the straight walks).

## 4. Tests and gates

Fast tests run in the default suite (`MUJOCO_GL=disable PYTHONPATH=. $PY -m pytest -p no:cacheprovider tests`);
slow ones behind `STAGE2_SLOW=1`, as `tests/test_stage2_look.py` does. A test that needs restored arrays or the
round-two model files skips without them (so it never makes `plant.model_files` write `stage1/models/`). Every
test that builds arrays does so in a subprocess with the three thread variables set to 1.

### 4.1 `tests/test_stage2_moves.py`

Exact equality with Stage 1:

| Test | Default | Asserts |
|---|---|---|
| `test_legacy_gaits_equal_stage1` | fast | for both `look.BODIES` and every name of `refs.reference_names(body)` plus `"stand"`: `moves.stage1_gait(body, name, H).legacy() == transitions.gait_of(body, name, H)` as dataclasses, the four in-place turn gaits included (speed 0.0) |
| `test_legacy_plans_equal_stage1` | fast (about 15 s) | for both bodies and all 18 `TRANSITIONS`: every array of `moves.plan(moves.stage1(body, name, H), geo)` is `array_equal` (NaN-aware) to `transitions.plan(name, a, b, geo)`'s, plus `out`, `ramps`, `stand_point`, `swings`; the gaits come through `GaitX.legacy()`, not from `gait_of` |
| `test_legacy_generate_equals_stage1_at_one_correction` | fast (4 solves each way) | with `transitions.gait_of` patched to return its gaits with `zmp_iters=1` (the patch reaches both generators, turn gaits included, and runs the correction once with each gait's gain): `moves.arrays(moves.generate(...))` and `transitions.arrays(transitions.generate(...))` have the same key list in the same order and every key `array_equal`, for `nominal_to_brisk` and `stop` (the `walk.zmp_iters if b.stand` branch) on `60-N-T-10`; `summary` equal on every key Stage 1 writes but `seconds` |
| `test_legacy_solver_is_stage1s` | fast | during a legacy `generate`, the functions called are `transitions.com_path`, `solve_frames`, `derivatives`, `multibody_zmp`, `idyn` themselves (wrapped counters), never the causal path |
| `test_legacy_generate_equals_stage1` | slow (about 15 min) | the same equality at the full 6 passes for `start, stop, brisk_to_fr05, enter_walk_0.50_left, stop_in_place_0.50_right` (exact split, an in-place gait), and each saved npz's sha256 equals `stage1/runs/2026-11-08/60-N-T-10/transitions.json`'s |
| `test_legacy_check_record_is_current` | fast, no arrays | `stage2/sets/legacy_check.json` exists, both picks `complete` with 18 of 18 `equal`, and its `moves_sha256` is the sha256 of today's `moves.py` (so an edit to `moves.py` fails the suite until the command is run again) |

The causal builder and the stride rule (no inverse kinematics, or one pass; all fast):

| Test | Asserts |
|---|---|
| `test_incremental_preview_equals_preview_run` | on one ZMP sequence, the incremental form's CoM equals `transitions.preview_run`'s to 1e-12 |
| `test_nothing_before_the_command_knows` | from one V0 0.40 walk, the CoM path and every plan array before the command frame are `array_equal` between the stop and the turn (two different commanded plans) |
| `test_null_command_reproduces_the_walk` | a "turn" of rate 0 and factor 1 (or a change from a walk to itself, built causally): every output frame within the G3 limits of the periodic walk (CoM 1.0 mm, ZMP 2.0 mm; one pass) |
| `test_rule_f_keeps_committed_footprints` | every stance planted before the command has the walk's footprint exactly (position and yaw differences 0.0); swings that lift at or after it land on the commanded path |
| `test_command_must_be_a_lift_off` | `plan` raises `ValueError` for a causal schedule with command phase 0.2; accepts 0.0, 0.5, -0.5 |
| `test_right_turn_mirrors_left` | plan level: foot positions, yaw, ZMP target and heading of the right turn are the left turn's mirrored, to 1e-12 |
| `test_turn_closes_and_ramps_last_0_375_s` | `kappa` within 1e-6, total heading within 1e-9 rad of pi/2; both ramps last 0.375 s of reference time for V0 (f 0.8) and F25F9cl (f 0.65); reference heading at the bar's 2.0 s and at the slowest clock at least 80 deg for both |
| `test_appendage_rule` | in a turn at f 0.8 the planned tail and neck amplitudes at the hold are 0.64 of the walk's; in the walk before the command they are the walk's |
| `test_causal_fade_window` | `tau_c` is zero before the command and from one stride after it; `post_window_zmp_error_m` is computed over `p >= c + 1` only |
| `test_pick_stride_factor` | with `kinematic` stubbed by a table: the largest passing f is chosen; both directions must pass; every f tried is recorded; no f -> `chosen: None`; the full-build step-down happens at most twice; the "over a leg drive" and "a1_over" outcomes are told apart by the caller's flags |
| `test_causal_join_a_is_null` | `summary` of a causal move has `join_a: None`, `stop: None`, `join_command` with `at_command.zmp_mm`, and no wall time |
| (slow) `test_stop_and_turn_from_v0` | a scratch build of `stop2_v0.4` and `turn90_pi4_right` from the V0 0.40 walk on `75-M-H-10` passes section 3.3's gates, with 2 touchdowns and the 80 deg rule |

### 4.2 Closed-loop bit-identity (`designlab/stage2/regress.py`, `stage2/regress/regress.json`, `tests/test_stage2_regress.py`)

The regression re-runs committed Stage 1 and Stage 1b episodes with the stock `eval` command of `tools/README.md`
("Eval traces") into a temporary `--out` and compares, per episode, (a) the trace's sha256 with the one in the
committed episode record and (b) the regenerated episode record with the committed one value for value, leaving
out only `wall_s`. (b) matters: the `control.py` edit touches `out["strides"]`, which no trace holds.

| Tier | Cases | Cost |
|---|---|---|
| fast | `75-M-H-10`, rows `R0,R2/42`, `start_stop/twice` and `walk/brisk`, seed 3042 (a hold, a stop event, a speed-change plan through `build_spec(plan=, gait=)`, mode R1 without terms, 0 ms); `60-N-T-10`, `--row-module designlab.stage1.speedfit --rows R2s/42@10ms,R2@10ms`, `turn/walking_0.35_left` and `turn/in_place_0.50_left`, seed 3042 (the 10 ms delay queue) | 8 episodes, about 4 core-min |
| full | fast, plus seed 3042 of every hard-floor condition of `75-M-H-10` and `60-N-T-10` under `R0` and `R2/42` from `stage1/runs/2026-11-08` (stand, walks, turns, start and stop, pushes), plus all 160 Stage 1b 10 ms turn episodes of `60-N-T-10` (the ones in `scratchpad/g9_traces`) | about 250 episodes, under 1 core-hour |

`regress run` clears `STAGE1B_SPEED_RUN` in the subprocess, refuses to run without the arrays and the round-two
model files, asserts `git status --porcelain` is empty before and after, and writes the record: `{tier, cases
[{run, body, row, condition, seed, trace_sha256, recorded, trace_equal, record_equal}], matched, of, sources
{panel_sources_sha256, stage1_sources_sha256 (plant.sources_sha256), tuner_sources_sha256, speed_sources_sha256},
versions, blas_threads, commit, dirty, at}`.

- `test_regress_record_is_current` (fast, no arrays): the committed record is of tier `full`, `matched == of`, and
  its four source hashes equal today's. No hash is a hand-written constant: after any Stage 1 edit (the
  `control.py` edit, the Jetson body's merge, a later round's) the suite fails until `regress run --tier full` has
  been run on the new code and its record committed. A constant bump proves nothing; a rerun does.
- `test_regress_fast` (slow): `regress run --tier fast` into a temporary record: 8 of 8.

It is an acceptance test of (a) the `control.py` edit, (b) the state-A freeze (the record is made again at the
freeze commit, ruling 8), (c) any later Stage 1 edit.

### 4.3 Gates on a set (`sets verify`; `tests/test_stage2_sets.py`, `tests/test_stage2_setio.py`)

`sets verify` (seconds; no array is rebuilt):

1. Every npz equals its entry's sha256; a frozen set's files and entries equal its latest manifest; every name
   parses back (`setio.parse_name`) to the values in its definition; no transition is named `stop` or
   `stop_in_place_*`; `blas_threads == 1` and `accuracy == "r2"` in every `built_by`.
2. Every move has `pass_` (section 3.3) or is recorded `available: false` with its numbers.
3. Stops: exactly two touchdowns after the command, stopped within 1.5 strides, 1 Hz acceleration within
   1.5 m/s^2. Turns: heading total within 1e-9 rad of +-pi/2; reference heading at bar time `(pi/2)/rate` at least
   80 deg on the nominal and on the slowest clock; hip-yaw use at most 0.87; the stride rule's record complete;
   `S2A-appendage` among the rules.
4. Walks: the Stage 1-format `q`, `qd`, `qdd` equal the look-format file's (`array_equal`; D3); the spec in the
   entry equals `SETS[set].spec_of(...)`; one crouch per set and body; `tuned_f9_spec` equals the PR's function at
   every rung (import by path). V0: `nominal` and `start` hash to Stage 1's; `look/nominal.npz` and `look/v0.4.npz`
   hash to the look ladder's V0 rungs.
5. Screen: a block for every available entry with `arrays.sha256` the entry's; `verdict.blocks_freeze` empty for
   every entry to be frozen; a turn with a neck or tail ratio above 1 carries `a1_over: true` **and** the
   `S2A-appendage` rule, else it is refused; V0's `nominal` and `v0.4` blocks equal `stage2/look/screen.json`'s
   rung blocks on every key but `speed`, `variant`, `seconds`, `arrays.file` and the added `range`.
6. Provenance: reports, without failing, "built by other code, arrays identical" when an entry's
   `built_by.stage2_modules` differ from today's files; fails on `mixed_sources` or `dirty` unless the manifest
   flags them.

Tests (fast on synthetic or tiny sets unless marked "built": those need the built 75 cm sets and skip without):

- names: every `setio` name round-trips exactly; `turn_name("pi4", ...)` maps to `math.pi / 4`; no name matches a
  fixed-decimal pattern or a three-significant-digit rounding;
- `entry_sha256` ignores `label`, rule texts, `look_sha256` and `built_by` and changes with `period_s` or
  `move.command_phase`; a bar key recomputed from another working directory is equal (`s2io.portable`);
- freeze: a second freeze after adding an entry keeps every earlier `(name, sha256)`; replacing a frozen entry's
  file or entry is refused; `sets build` on a frozen set with a missing array restores it only when the rebuilt
  sha256 equals the manifest's and never rewrites a committed JSON (mtimes unchanged);
- resume: `sets build` on a complete set runs nothing and rewrites no file; killed between every pair of writes
  (npz placed, partial written, JSON merged, partial deleted), a rerun ends in the same files, with `git status
  --porcelain` empty but for JSON and Markdown; a second builder on a locked directory is refused; a stale lock is
  taken over;
- process state (built, slow): the same walk built first in a fresh process and built after a move in one
  process has the same sha256, and with `--procs 1` and `--procs 2`;
- ignore rules: `git check-ignore` on one path of every output kind and on a temporary name of every writer
  (`refs/x.npz`, `refs/x.npz.tmp`, `refs.json.tmp`, `partial/x.json`, `.lock`, a trace, `logs/x.log`, `done/x`);
- no-op edit: appending a comment to a copy of `moves.py` leaves `sets verify` passing with the "other code" note
  and changes no bar key;
- (built) `test_v0_set_reuses_stage1_files`, `test_v0_look_files_are_the_ladders`.

### 4.4 Readings (`tests/test_stage2_readings.py`, all fast, no MuJoCo) and the real-data check

| Test | Construction | Asserts |
|---|---|---|
| `test_own_swing_clearance_ignores_a_delay` | a foot trajectory (sin^2 lift of 0.081 H, swing 0.350 s) and its contact flags; the simulated foot and contact are the reference's delayed by k = 0..5 physics steps while `ref_swing` stays on the reference's clock | `toe_clearance_own_swing_m` equal for every k within 1e-6 m; the reference-clock reading is above 0.04 H at k = 0 and below it at k = 5 ([review] 0.0422 falling to 0.0351 H) |
| `test_own_swing_edges_are_interpolated` | the same swing with the lift-off moved by 0, 2, 4, 6, 8 ms across the 10 ms grid | the reading moves by less than 1e-6 m (rows alone: up to 2.4 mm) |
| `test_early_touchdown_fails` | a touchdown at 45 % of the planned swing | the air run is a swing (it overlaps the reference's), read over its own middle half; the term fails |
| `test_mid_swing_stub_fails` | a 30 ms re-contact at mid-swing | two swings in one reference swing: `interrupted_swings == 1`; the term fails |
| `test_stance_chatter_is_an_air_gap` | a 30 ms unloading in mid stance | in `air_gaps`, not read; the clearance unchanged; the same unloading lasting more than `air_gap_max_s` is read as a swing |
| `test_swing_count` | a window in which one foot never lifts | `own_swings < ref_swings - 1`: the term fails; a swing cut by the window's end is not read and does not fail it |
| `test_line_reading` | a CoM path of 4.2 m with a 0.2 m bow, with a 0.3 m bow, 3.9 m straight; a heading of 30 deg | along and lateral to 1e-12; `line` true, false; `along` false; anchoring at the window's first update; the episode-start line reported |
| `test_turn_reading` | yaw reaching 82 deg at `at_s` and 90 after; one reaching 79; one overshooting to 101 deg; a right turn; a yaw crossing +-pi; a 0.5 rad/s turn read at 3.1416 s | `heading` true, false, false; `time_to_within_10deg_s`; the sign; no wrap error; `at_s` per rate |
| `test_stop_reading_counts_touchdowns` | contact sequences with 2 and 3 touchdowns after the command, one exactly at the command step, a 10 ms bounce, a late landing of the earlier step | 2, 3; the one at the command not counted; the bounce merged by the debounce; the late landing counted and visible in `last_touchdown_before_command`; `stopped_by_end` false when the speed never settles |
| `test_stop_acceleration_bound` | a step change of 0.42 m/s in forward speed | the 1 Hz reading is at most 0.93 m/s^2; the raw peak is reported |
| `test_drive_ratios_split` | `force_sq` in which only the tail exceeds its sustained rating | `drive_sustained` false, `drive_sustained_legs` true; worst leg and worst A1 drive named |
| `test_lowpass_equals_stage1s` (may import `stage1.metrics`) | random series | `readings.lowpass == metrics.lowpass` |
| `test_reference_clock` | a transition-format array with a known heading law | `reference_heading_at(arr, 2.0, 1.05)` reads the heading at 2.1 s of its own time; at rate 0.9975 at 1.995 s |

**The real-data check (G9's toe-clearance evidence; WP B's acceptance, not a pytest).** `bar reread` re-runs a
sample of the Stage 1b 10 ms turn episodes of `60-N-T-10` (seed 3042 to 3046 of the four gated turn conditions
under `R2s/42@10ms` and `R2@10ms`: 40 episodes, about 6 core-min) with the stock rows and conditions. For each it
asserts the stock trace's sha256 equals the committed episode record's (the 160 traces in `scratchpad/g9_traces`
are that baseline), then reads (a) on the live recorder. The output record lists, per episode, the own-swing and
the reference-clock readings, their difference, interrupted swings and air gaps; it is cited as G9's
toe-clearance evidence. The full 160 are run with the same command before G9 is signed. `bar refcheck` covers
G9's other half for the sets: every frozen reference played kinematically through `bar.reference_recorder`
passes its own bar item (2 touchdowns, 80 deg at `at_s`, own-swing clearance at least 0.04 H, 4.0 m in 10 s), and
`R0@10ms` episodes carry every reading (checked in `test_stage2_bar.py`'s slow episode).

### 4.5 `tests/test_stage2_setscreen.py`

- (built) the free regression: V0's `nominal` and `v0.4` blocks equal `stage2/look/screen.json`'s (4.3 item 5);
- `screen_move` on Stage 1's `start` and `enter_walk_0.50_left` of `75-M-H-10` reproduces `transitions.json`'s
  recorded single-support CoP margin and friction ratio, and its drive ratios equal `envelope._ratios` on the
  file's `tau` and `omega` (needs the Stage 1 arrays);
- positive controls: a move with its lift scaled to 0.03 H fails the swing gate; `omega` scaled by 1.3 reads
  "over" on a leg drive and blocks; a turn at 0.83 of the twist passes its range gate and fails Stage 0's 0.10 rad
  gate, which is not applied to it; a turn with the tail's torque scaled over 1 gets `a1_over` and does not block;
- `clearance.screen` is called with `substeps = 0`; a window that ends inside a swing is refused;
- the block has no wall time; `markdown(doc)` regenerates `SCREEN.md` byte for byte;
- `tests/test_stage2_screen.py`, `test_stage2_look.py`, `test_stage2_motion.py` pass unchanged and `git diff
  --stat stage2/look` is empty.

### 4.6 `tests/test_stage2_bar.py`, `tests/test_stage2_barreport.py`

Fast (stub set: `stage1/runs/2026-11-08/<body>` through `setio.stage1_view`, or a tiny synthetic set):
the 16 gated conditions and their keys; plans' `seconds` and track lengths obey section 3.8's table at the
slowest and the fastest clock; `Delayed(row, 0)` on a short stock walk equals `panels.TrackerRow`'s episode value
for value; `Delayed(R0, 5)` equals `speedfit.servo_delay` applied by hand; `Delayed` wraps a
`policy_rows.PolicyOnlyRow` (constructed with a stub policy) and merges its sources; row names parse
(`R2/42@4ms`, `R0@10ms`; `R2/42@3ms` and `@22ms` raise); the key ignores a label and another entry and changes
with a file's sha256, `bar.py`'s bytes or a row source; **a status-done record with another key is refused and
left untouched; `--supersede` moves it aside and runs**; resume skips a complete record, reuses partial episodes
of the same key only, reruns an episode with `error`, and a torn partial file is ignored; kill and resume gives
records equal but for `wall_s`, `started_at`, `ended_at` and `sessions`; a second runner on a locked run directory
is refused; a spawned worker (`--procs 2` smoke) refuses other `bar_sources` and a thread count other than 1; a
raising episode becomes `failed_episode` in the worker; an incomplete judged window is an error, not a failed
term; the two readings' counts differ in a synthetic record where only the tail is over; a "not run" record is
written for a condition whose reference is `available: false`; a nudge's force is `m x 0.25 x bound / 0.1`; the
stop's event reaches `control.run`'s `stops`; the ledger refuses a tier that does not fit; the replication block
is judged on its own seeds and never pooled; stock (`S1`) conditions are judged by `panels.judge`. Slow: one full
episode per judge on a built `75-M-H-10` set under `R0@10ms` with every reading present.
`barreport`: tables regenerate byte for byte from records; readings in the plan's order; blocks side by side;
"not run" and "cut by the pre-registered order" cells; no measured value in a target column; JPS's numbers only
in columns labelled as its claims.

### 4.7 What a reviewer can run, and what an independent reviewer of the finished build should verify

```
export PYTHONPATH=. MUJOCO_GL=disable OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1; PY=.venv/bin/python
$PY -m pytest -p no:cacheprovider tests/test_stage2_s2io.py tests/test_stage2_setio.py tests/test_stage2_readings.py \
    tests/test_stage2_moves.py tests/test_stage2_sets.py tests/test_stage2_setscreen.py tests/test_stage2_bar.py \
    tests/test_stage2_barreport.py tests/test_stage2_regress.py                                  # fast set: about 6 min
$PY -m designlab.stage2.moves legacy --body 75-M-H-10 --name stop --name brisk_to_fr05           # 2 min; exit 0
$PY -m designlab.stage2.regress check                                                            # the record against today's code
$PY -m designlab.stage2.sets verify --set F25F9cl --body 75-M-H-10                               # seconds
$PY -m designlab.stage2.bar dry-run --set F25F9cl --body 75-M-H-10 --rows R2/42@10ms --out stage2/runs/<run>   # writes nothing
git status --porcelain; git check-ignore -v stage2/sets/F25F9cl/75-M-H-10/transitions/stop2_v0.4.npz
```

The independent reviewer's list:

1. `moves legacy` on both bodies gives 36 of 36 sha256; the legacy path calls `transitions`' own functions; no
   Stage 1 file but `control.py` and no committed look file changed (`git diff --stat` on `designlab/stage1`,
   `stage1/`, `stage2/look`); `regress.json` is current and of tier full.
2. V0's `nominal` and `start` hash to Stage 1's; V0's look-format `nominal` and `v0.4` hash to the ladder's V0
   rungs with equal screen blocks.
3. Causality on the frozen stop and turns: frames before the command equal the walk, the null command reproduces
   the walk, left and right mirror, rule F holds, the fade window is as recorded.
4. The stride rule's pick recomputes from the recorded tries and its constants match the pre-registered text; no
   turn is frozen against the leg-drive rule; every `a1_over` turn carries the appendage rule.
5. From raw traces of a sample of episodes that includes failures: the own-swing clearance, the line reading, the
   heading error at the bar's clock and the touchdown count recompute to the records' values; the command instant
   equals `sim_time` of the move's first frame.
6. The rows are unchanged: gains equal `tune.json` with its sha256; k = 0 equals `TrackerRow` bit for bit; the
   delay queue equals `speedfit`'s; `R0@0ms` equals the panels' R0.
7. Record keys recompute; a kill and resume leaves equal records; `git status --porcelain` shows only JSON and
   Markdown and every array path is ignored.
8. Counts, bounds and verdicts per reading (every limit; but for A1); the judged delay is 10 ms; the replication
   block is labelled and never pooled; every F25F9cl record carries the margin label.
9. The tables regenerate byte for byte; no measured value sits against a target; "not run" and "cut" cells are
   shown; JPS's numbers appear only as claims.
10. Every frozen reference has a screen block with each gate; dropped rungs and absent turns are recorded with
    their numbers.
11. Core-hours in `budget.json` against the plan, and that what was cut is the tail of the pre-registered order.

## 5. Work packages

Each builder works in its own git worktree on its own branch; no two packages own the same file. A workflow runs
at most two agents at once on this machine (`LESSONS_LEARNED.md` 4.2), and the machine is shared with the restore
job, so the packages run **two at a time, each with one heavy process at a time** (`--procs 1` while two agents
build). Briefs: `$D/WP_<name>.md`.

| Wave | WP | Owns (exclusively) | Needs | Compute cap |
|---|---|---|---|---|
| 1 | **M** moves | `designlab/stage2/moves.py`, `tests/test_stage2_moves.py`, `stage2/sets/legacy_check.json` | Stage 1 arrays of both picks in its worktree | 1.5 core-h (the legacy check is 0.7) |
| 1 | **R** readings and shared I/O | `designlab/stage2/readings.py`, `s2io.py`, `setio.py`, their three test files, `stage2/sets/.gitignore`, `stage2/runs/.gitignore` | none | minutes |
| 2 | **G** guard | `designlab/stage2/regress.py`, `tests/test_stage2_regress.py`, `stage2/regress/regress.json`, the edit to `designlab/stage1/control.py` | Stage 1 arrays, round-two models; R merged | 2 core-h (the full tier twice) |
| 2 | **C** screen | `designlab/stage2/setscreen.py`, `tests/test_stage2_setscreen.py`, the one line of `designlab/stage2/screen.py` | R merged; Stage 1 arrays and the look ladder's V0 arrays | 0.5 core-h |
| 3 | **S** sets | `designlab/stage2/sets.py`, `tests/test_stage2_sets.py`, the edits to `tools/restore_inputs.py`, `tools/README.md`, `tests/test_restore_inputs.py` | M, R, C merged | 1.5 core-h (V0 built once; the three sets are the lead's run) |
| 3 | **B** bar | `designlab/stage2/bar.py`, `tests/test_stage2_bar.py`, `stage2/runs/2026-11-15-75cm/run_75cm.sh` | R merged (it starts on the `S1` view); S's V0 for its smoke | 1 core-h |
| 4 | **T** tables | `designlab/stage2/barreport.py`, `tests/test_stage2_barreport.py` | B merged | minutes |
| later | **L** look | `designlab/stage2/moves_look.py`, `tests/test_stage2_moves_look.py`, and `sets.py`'s `SETS` entries for V6 and V6noF5 (after S is merged) | M, S merged; state A frozen | separate budget |

Dependencies that are easy to miss: S's `verify` and `freeze` read C's `screen.json` blocks (the keys listed at the
end of section 3.7); B's `bar_sources` hashes R's `readings.py` and `setio.py`, so R's files are frozen with state
A; G's record must be made again by the lead at the freeze commit, after the Jetson body's merge (ruling 8); L
never edits `moves.py`: the `Laws`, `PlanCtx` signatures and the `solver == "look"` dispatch are M's, written now
(section 3.9), and if L finds a signature wrong it asks the lead, and any change to `moves.py` then needs
`legacy_check.json` and every frozen set's `sets verify` to pass again.

Worktrees have none of the git-ignored inputs (`LESSONS_LEARNED.md` 4.3). Each brief starts with:
`$PY tools/restore_inputs.py copy-from /home/user/mesozoic-labs/hardware/trex_form_study --group picks-models
picks-arrays` (and `look` for C), which copies and checks every sha256; the link or copy directories are
excluded through `.git/info/exclude` if they show as untracked.

Public names frozen by this document (a builder who needs another signature asks the lead): M's `GaitX, Schedule,
Move, Laws, PlanCtx, TODAY, plan, generate, kinematic, arrays, summary, pick_stride_factor, legacy_check, start,
change, stop_one_step, stop_two_step, turn, stage1, stage1_gait`; R's every function of section 2.1 and the
readings of 2.3; the entry and record schemas of sections 3.6 and 3.8.

## 6. Run plan

All from the study root, `PY=.venv/bin/python`, `export PYTHONPATH=. MUJOCO_GL=disable`.
`RUN=stage2/runs/2026-11-15-75cm` (records, `budget.json`, `run_75cm.sh`, `COST.md` committed; `traces/`,
`partial/`, `logs/`, `done/`, `*.npz`, `*.tmp`, `.lock` ignored).

**The chain.** `$RUN/run_75cm.sh` is one idempotent script (the pattern of
`stage1/runs/2026-11-10-speed/run_speed_round.sh`): each step is `step <name> <command...>`, which skips when
`$RUN/done/<name>` exists, runs the command through `tools/lowprio.sh`, appends the command's own exit code to
`$RUN/logs/<name>.log`, and touches the marker only on exit 0. Every command is itself resumable (a fresh clone
without markers re-walks the steps and finds every finished record). The watchdog entry
(`tools/watchdog_example.sh`'s form, run by `tools/keepalive.sh` every 10 min) starts the chain when
`$RUN/done/ALL` is absent and `pgrep -f run_75cm.sh` finds nothing; `bar`'s run-directory lock stops a second
copy. Check `uptime` and `df -h .` before starting; two processes (`--procs 2`).

```
# --- before state A (the lead; WPs merged, Jetson body merged) ---------------------------------------------------
gate legacy-75    $PY -m designlab.stage2.moves legacy --body 75-M-H-10 --procs 2 --record stage2/sets/legacy_check.json
gate legacy-60    $PY -m designlab.stage2.moves legacy --body 60-N-T-10 --procs 2 --record stage2/sets/legacy_check.json
gate regress      $PY -m designlab.stage2.regress run --tier full --record stage2/regress/regress.json
for S in V0 F25F9 F25F9cl; do
  step build-$S   $PY -m designlab.stage2.sets build --set $S --body 75-M-H-10 --procs 2
  step screen-$S  $PY -m designlab.stage2.setscreen run --set $S --body 75-M-H-10 --procs 2
  step md-$S      $PY -m designlab.stage2.setscreen md --set $S --body 75-M-H-10
  step verify-$S  $PY -m designlab.stage2.sets verify --set $S --body 75-M-H-10
done
# -> the lead reads the three SCREEN.md files (stride factor and twist of each turn, a1_over, grip need, dropped
#    rungs, "not available" items), commits, then freezes:  sets freeze --set $S --body 75-M-H-10   (state A)
step refcheck     $PY -m designlab.stage2.bar refcheck --set V0 --body 75-M-H-10 ... (each set)
step cost         $PY -m designlab.stage2.bar dry-run ... > $RUN/COST.md        # measured rates; committed before T3.1

# --- task 3, the walk: ledger "3", 12 core-h. W="walk/v0.4,walk/v0.4_mu0.54"; rows(d) = R2/42@d, R2/43@d, R0@d ----
T3.1   bar run  W            @10ms  sets F25F9cl, F25F9, V0     --attribute judged       # the judged delay first
T3.2   bar run  --block seeds5042 --passed-of seeds3042  F25F9cl  W @10ms                # replication of what passed
T3.3   bar run  W            @0ms   F25F9cl, F25F9, V0
T3.4   bar run  W            @4ms then @6ms
T3.5   bar run  W            @2ms then @8ms
T3.6   bar run  speed ladder (walk/nominal, v0.3, v0.35, v0.45, v0.5), mu 0.9, R2 rows, F25F9cl then V0: @10ms, then @0ms
T3.7   bar run  JPS rows @0ms: F25F9 walk/v0.5; F25F9cl walk/v0.45_mu0.54, walk/v0.5_mu0.54;  @2ms: F25F9cl walk/v0.5
T3.8   bar run  --block seeds5042 --passed-of seeds3042  F25F9cl  everything of T3.3-T3.7
T3.9   bar run  speed ladder @4ms then @6ms
T3.10  bar run  --attribute all   (the records of T3.3-T3.9 with 5 or more falls, in that order)

# --- task 4, the rest of the bar: ledger "4", 12 core-h. B = the 14 turn, nudge and stop conditions ---------------
T4.1   bar run  B            @10ms  F25F9cl then V0             --attribute judged
T4.2   bar run  --block seeds5042 --passed-of seeds3042  F25F9cl  B @10ms
T4.3   bar run  B            @0ms   F25F9cl then V0
T4.4   bar run  B            @4ms   F25F9cl, then V0
T4.5   bar run  B            @6ms   F25F9cl, then V0
T4.6   bar run  reported ladders @10ms, mu 0.9, F25F9cl then V0: turn/t90_{0.35,0.5,1}_{left,right},
                nudge/<dir>_x0.375, _x0.5, stop/one_step
T4.7   bar run  --block seeds5042 --passed-of seeds3042  F25F9cl  B @0ms, @4ms, @6ms
step report  $PY -m designlab.stage2.barreport --out $RUN       # BAR_TABLES.md, bar_tables.json, the JPS 2.3 table
```

Each `T` line is one `step` per set and delay (one command per delay tier, so a budget stop cuts whole delays in
the lead's order 10, 0, 4, 6, 2, 8 ms and never half a tier). The ledger rule (D22) is applied before every
step and before every record's attribution; R0 rows run with `--no-traces` if `df` shows under 1.5 GB free.

Estimated cost, to be replaced by `COST.md`'s measured rates before T3.1 ([review] R2 0.39-0.47 s of wall time
per simulated second, taken as 0.45; R0 0.08; V0 episodes 18.9-21.1 s, F25 sets 23.5-24.4 s; so about 9 and 11
core-seconds per R2 episode):

| Step | Episodes | Core-hours |
|---|---|---|
| gates: legacy check, two bodies; regression, full tier | - | 0.7 + 1 |
| three sets with screens (6 walks in two formats, start, 5 changes, 2 stops, the stride rule and up to 8 turns each) | - | about 3 |
| T3.1 walk at 10 ms: 3 sets x 2 grips x (2 R2 rows + R0) x 20 | 240 + 120 | 0.7 |
| T3.1 attribution: up to 12 records x 3-4 lifts x 20 | up to 960 | typically under 1; worst case 2.7 |
| T3.2 replication at 10 ms | up to 80 | up to 0.25 |
| T3.3-T3.5 walk at 0, 4, 6, 2, 8 ms | 1,200 + 600 | 3.7 |
| T3.6 ladder at 10 and 0 ms: 2 sets x 5 rungs x 2 rows x 20, twice | 800 | 2.2 |
| T3.7 JPS rows | 160 | 0.45 |
| T3.8 replication of T3.3-T3.7 | up to 880 | up to 2.5 |
| T3.9 ladder at 4 and 6 ms | 800 | 2.2 |
| T3.10 attribution of the rest | up to about 2,000 | up to 5 |
| T4.1 bar at 10 ms: 2 sets x 14 conditions x (2 R2 rows + R0) x 20 | 1,120 + 560 | 3.4 |
| T4.1 attribution (records with 5 or more falls after the command) | up to 4 lifts x 20 per record | 0.2 per record |
| T4.2 replication at 10 ms | up to 560 | up to 1.7 |
| T4.3 bar at 0 ms | 1,120 + 560 | 3.4 |
| T4.4, T4.5 bar at 4 and 6 ms, candidate then V0 | 2 x (560 + 280), twice | 1.85 + 1.5 each |
| T4.6 reported ladders | 2 x 600 | 1.8 + 1.5 |

Read honestly: task 3 reaches T3.7 inside 12 core-hours in the typical case (about 8) and T3.8-T3.9 only in
part; task 4 reaches T4.3 (about 8.5 with replication, before attribution) and then has room for the candidate's
4 ms tier at most. Everything not started is reported as cut, by name. Disk: about 90 kB of trace per episode,
0.6-0.8 GB for both tasks against about 4 GB free.

On-disk layout of the outputs:

```
stage2/sets/<set>/75-M-H-10/{refs,transitions,look,partial}/, refs.json, transitions.json, set.json, screen.json, SCREEN.md
stage2/runs/2026-11-15-75cm/<set>/75-M-H-10/<block>/eval/<row>/<behavior>/<condition>.json     e.g. .../seeds3042/eval/R2/42@10ms/turn/q90_left_mu0.54.json
stage2/runs/2026-11-15-75cm/<set>/75-M-H-10/<block>/eval/traces/<row>/<behavior>/<condition>/<seed>.npz
stage2/runs/2026-11-15-75cm/<set>/75-M-H-10/<block>/partial/<row>/<behavior>/<condition>/<key hash>/<seed>.json
stage2/runs/2026-11-15-75cm/<set>/75-M-H-10/<block>/eval/summary.json
stage2/runs/2026-11-15-75cm/{run_75cm.sh, budget.json, COST.md, BAR_TABLES.md, bar_tables.json, logs/, done/}
```

## 7. Risks and open questions for the lead, most important first

1. **Two definitions differ from ruling 5, on reviewers' evidence; please confirm or overrule before the
   pre-registration.** (a) *Air gaps.* The ruled "an air run shorter than half the shortest planned swing is an air
   gap" lets an early touchdown or a mid-swing stub leave the judged clearance reading. This design keeps the
   threshold only for air runs wholly inside the reference's stance, reads every air run that overlaps a reference
   swing, and fails an interrupted swing (section 3.8 a). (b) *Turn ramps.* The ruled half stride gives the
   candidate a 0.410 s ramp against the control's 0.387 s and a reference at 79.7 deg at 2.0 s of its own clock.
   This design uses 0.375 s for every set (D6). If the lead keeps half a stride, the candidate's reference still
   passes the bar-clock rule (84.2 deg) but not the slowest-clock margin, and that gate must be dropped for it.
   The prototypes ran at 0.387 s; 0.375 s has been run only at plan and kinematic level (E6b).
2. **The budgets do not hold everything asked.** With the replication ruled for every passed condition and
   attribution on every record with 5 falls, task 3's worst case is about 20 core-hours and task 4's more. Section
   6's order (judged delay, its attribution, its replication, then 0, 4, 6, 2, 8 ms, ladders, the rest) and the
   ledger rule decide what is cut; the likely losses are task 4's 6 ms tier, V0's 4 ms tier, the reported turn
   and nudge ladders, and attribution of non-judged delays. The lead should confirm that order, or give a larger
   budget, before T3.1.
3. **On this body the control cannot pass reading 2 at 0.35 m/s and above** (the tail servo fails
   `drive_sustained` in 20 of 20 Stage 1 episodes, V5). The 75 cm verdicts will therefore mostly be decided in
   reading 3, and D13 lets either reading trigger the replication. The headline stays reading 2 with reading 3
   beside it, as the plan orders them; the lead may want that sentence in the pre-registration's own words.
4. **The candidate may have no quick turn, or only one read under reading 3.** F25F9cl needs stride x0.65 inside
   the turn; [review] its kinematic hip-pitch speed ratio is 0.968, so a full build may pass 1.0 and D7 then
   freezes no reference (to the owner). On `60-N-T-10` the lead's steady-turn pricing is over on three leg speeds
   at every stride tried. Neither turn has been built in full; ruling 1's amplitude rule has not been priced.
5. **The one Stage 1 edit moves to before state A** (D18), against the draft and ruling 7's spirit that the look
   never blocks the 75 cm work. It is three lines with a default-identical path and a one-hour regression, and it
   buys one panels hash for all of Stage 2. The alternative is no Stage 1 edit in round A and the look's K-stride
   walks played as finite segments (cut 4 of the draft), which fixes R5's walks to whole windows.
6. **Every item shares the walk's lead-in and seed slot** (D12). It makes the eight items of a seed dependent and
   is the reason attribution cites the walk record. Stage 1 gave turns their own slots; if the lead prefers that,
   it is one table in `bar.py`, before any run.
7. **Causal references make the bar harder than Stage 1's convention** (ruling 2 accepted this): the stop reads
   about 1.2 strides and two touchdowns, and every turn starts with a 13-28 mm ZMP transient that halves the
   single-support CoP margin (24 against 45 mm). The solver is new code; section 4.1's unit tests and the fade
   window (V1) are its protection.
8. **`grf` is reported, not judged, in every item from 0.40 m/s** (Stage 1's own rule for non-nominal walks,
   extended to the turn so that the walk and the turn carry the same limits). The lead may prefer it judged in
   both.
9. **Thigh-twist margin in closed loop.** 0.85 rests on R2 holding decision 14 at 0.847 on this body at 0 ms. At
   10 ms it may not hold.
10. **JPS 2.3 cells this plan does not fill:** 0.55 m/s (outside the ladder), the Jetson-mass row (task 6), 8 ms
    at 0.50 m/s, the stop from 0.50 m/s and at 2 ms, nudges at 2 ms. `barreport` prints them as "not run". JPS's
    0.496 m/s is set beside this plan's 0.50 m/s rung, stated in the table.
11. **Track frequency** (D16): an optional reported control, the candidate's judged 10 ms walk with omega at
    0.03 H on the same seeds (40 episodes, 0.12 core-h), if the lead wants the sensitivity on record.
12. **The look.** J1's resampling is the hard part of F14 in moves; the boundary-only rule is the named cut.
    L10 (flat-footed commanded moves) is the pre-registered fallback for F5 and is what R2L's set needs anyway.
13. **Smaller.** The recorder's one-step timing convention is checked once by WP B (section 3.8). `bar` must
    release a worker's cached plants and tracks when the set changes (Stage 1's panels keep every model for the
    life of the process, lesson 12). The `sets` restore group rebuilds about 3 core-hours of arrays in a fresh
    clone.

**For the owner** (through the lead): ruling 1's tail and neck rule, told as a look choice; lever 6 or lever 1 if
D7 freezes no quick turn for a set or body; nothing else in round A needs an owner decision.

## Review log

Each finding was checked against the code (file and lines read are named where the check was not a plain reading
of the cited lines). "Accepted" means the design changed as stated; "in part" says what was not taken and why.

Provenance lens:

- **B1** accepted. Checked `look.py:281-289, 470-480` (the accuracy context restores the previous level) and `gait.py:300-312`. D3's process-state contract (`s2io.r2()` around the whole item, inverse dynamics inside it); entries record `built_by.accuracy`; `sets verify` compares `q, qd, qdd` of the two formats; a slow test builds one walk in two process states.
- **B2** accepted, item 4 in part. Checked `panels.py:1856-1888`. D14: the key holds only what an episode executes (`bar_sources` = `bar.py`, `readings.py`, `setio.py`, `speedfit.py`), named entry subsets, and a status-done record is never overwritten (`--supersede` or a new `--out`); tests in 4.6. A separate judge key is rejected (a re-judge needs about 0.5 MB of recorder per episode, 5 GB); traces carry contact bits and 100 Hz low points instead.
- **B3** accepted. Checked `control.py:967-978, 2300`. D18: the `control.py` edit lands before state A in WP G with the full regression; later edits to a `panel_sources` module re-run baselines into a new `--out` (pre-registered). Raised to the lead as open question 5.
- **B4** accepted. Identity (`definition` + file sha256) is separate from provenance (`built_by`, per entry); `moves_sha256` left the key; walk entries have a `definition`; `built_by` leaves the set's own outputs out of `dirty_paths`; `freeze` refuses mixed sources or a dirty tree unless flagged (3.6, D14, D20).
- **B5** accepted. D20: append-only manifest by `(name, sha256)`, `freezes[]` history, `sets build` on a frozen set only places matching arrays, records carry the freeze's manifest sha256, `barreport` checks inputs are still listed.
- **B6** accepted. Checked `transitions.py:165-179`. `GaitX.in_place`; `legacy()` is `gait_of`'s constructor call; `test_legacy_gaits_equal_stage1` runs before any plan is compared; E1's limit is stated in section 0.
- **B7** accepted. 4.2: the four 75 cm episodes join the fast tier; the full tier adds seed 3042 of every hard-floor condition of both picks under R0 and R2/42; the pinned constants are replaced by the committed `regress.json`, whose hashes the fast test compares with today's.
- **B8** accepted. Checked `look.py:119-123`. D19 and 2: variables set at the top of each CLI module before numpy, `s2io.require_one_thread()` refuses any other value in the parent and in every worker; `blas_threads` recorded per entry and session; `sets verify` gates on 1.
- **B9** accepted. `*.tmp` and `.lock` in both new `.gitignore` files; JSON through `<name>.json.tmp`, arrays through a file object on `<name>.npz.tmp`; stale temporaries swept under the lock; 4.3's ignore test includes temporary names and a killed build.
- **B10** accepted. D19: bar resumes per episode (worker-written partials keyed by the record's key hash, torn files ignored, sessions listed); sets per item with the committed JSON as the only truth; one lock per directory; kill points tested.
- **B11** accepted. Section 2's module skeleton (the `__main__` guard calls the named module's `main`); `Move` carries `builder` and `args` and workers rebuild it from `(set, body, item)`; a `--procs 2` smoke in 4.6.
- **B12** accepted. D4: `stage2/sets/legacy_check.json`, required by `sets freeze` and by a fast test that compares its `moves_sha256` with today's file.
- **B13** accepted. Checked `tests/test_restore_inputs.py:79-96`. 2.4: the `sets` group, its `NEEDS`, the pinned tuple, the file count and the README change in one commit, owned by WP S and due at the freeze.
- **B14** accepted. D3 and 4.3 items 4-5: V0's look-format `nominal` and `v0.4` must hash to the ladder's V0 rungs and their screen blocks must equal the committed ones.
- **B15** accepted. Checked `screen.py:495`. 3.7: `screen_walk` adds a `range` block from the look entry's margins and drops `seconds`; the wrong claim is gone.
- **B16** accepted. D2: the quick turn is `turn90_pi4_<dir>` from an exact table; paths anchored at `s2io.STUDY`; keys built from `s2io.portable(describe())` and tested from another directory; the regression clears `STAGE1B_SPEED_RUN`.

Dynamics lens:

- **MC-1** accepted; it amends ruling 5 (open question 1). Checked `metrics.py:160-178` (`ref_swing` per update). 3.8 a: an air run that overlaps the reference's swing is read whatever its length; the ruled threshold applies only to runs wholly in the reference's stance; an interrupted reference swing fails; `own_swings >= ref_swings - 1`; durations on one clock; two new tests.
- **MC-2** accepted; it amends ruling 5 (open question 1). Re-ran the kinematic pass at three ramp lengths (E6b): the twist is insensitive. D6: ramps of 0.375 s of reference time for every set; D9 and 3.4 gate the reference on the nominal and on the slowest clock; both sets' numbers are in section 0.
- **MC-3** accepted. 3.2 step 5: the excused transient is faded to zero over phases [c + 0.5, c + 1.0]; `post_window_zmp_error_m` gated at 4.0 mm; the window is in the move's rules.
- **MC-4** accepted. 3.9 J1-J3: a move leaves a K-stride window at the warp's own time and the join frames are the window resampled on the move's phase grid; plan items count windows and planners use `window_s`; the boundary-only rule is the named cut.
- **MC-5** accepted. 3.9 L10 is restated per stance (every stance ending after the command is flat; the swing lifting at the command carries the pitch to zero), with a causality test.
- **MC-6** accepted. Checked `transitions.py:719-757, 883-907`. `join_a` is `null` for a causal move; `join_command.at_command.zmp_mm` is recorded with its note and not gated (the reason is stated).
- **MC-7** accepted. 3.1 item 4 and D5: a causal command must be a lift-off; `plan` raises otherwise; `stop_two_step` takes no command argument; test with phase 0.2.
- **MC-8** accepted. Checked `tune.py:608-640` and `panels.NOMINAL_ONLY_TERMS`. 3.8 b: `stage1` holds the success dict restricted to `panels.terms_of`'s list with `succeeded` recomputed; `grf` is reported, not judged, in every item from 0.40 m/s (open question 8).
- **MC-9** accepted. 3.5 states the bound (0.93 m/s^2) and the margin before the command; 3.8 e records each counted touchdown, the last one before the command and the raw deceleration peak.
- **MC-10** accepted. Checked `control.py:154` (`COP_INSET = 0.005`). 3.3: causal moves need a single-support CoP margin of at least +10 mm, recorded with frame and foot; Stage 0's -10 mm stays the screen's flag.
- **MC-11** accepted. Every episode records `segment_reached`, `fell_before_command`, `window_reached`; reports count lead-in falls apart; both line anchors are kept (ruling 5 picks the judged one).
- **MC-12** accepted. Checked `transitions.py:176, 648-649`. The fast test patches `transitions.gait_of` and uses `nominal_to_brisk` and `stop`; the turn entry stays in the slow test and the CLI gate.
- **MC-13** accepted. 4.4: `bar reread` re-runs Stage 1b turn episodes with the stock rows, checks each stock trace's sha256 against the committed record and reads (a) on the live recorder; the result is G9's toe-clearance evidence. Only the trace identity is asserted; the comparison of the two readings is reported (own-swing need not exceed the reference-clock reading in every episode).

Scope lens:

- **COV-01** (blocker) accepted. Checked `metrics.py:254-256`, `panels.py:1084-1087`, `plant.py:205-208`. D10 and 3.8: per-drive ratios for both spans, `success` and `success_but_A1` with their own counts and verdicts, a reserved `look` reading, a synthetic tail-only test; in place before any panel episode.
- **COV-02** accepted. D13: "passed" is one record (row, condition, delay) in reading 2 or 3, replicated on the same row; section 6 has replication steps after each tier with worst-case costs; reports never pool the blocks. The budget consequence is open question 2.
- **COV-03** accepted. 4.1's second table (incremental preview, nothing before the command, null command, rule F, mirror, the stride rule's logic); the rules text says where the gate constants were set.
- **COV-04** accepted. 3.8 a: the reading includes `low` interpolated at the two edges of the middle half; the test uses a lift of 0.081 H with the reviewer's numbers; the equality tolerance is 1e-6 m (linear interpolation between 100 Hz rows of a curved lift is not exact to 1e-9).
- **COV-05** accepted. Same changes as B2 and B4; set definitions stay in `sets.py`, which is in no key; the no-op-edit test is in 4.3.
- **COV-06** accepted. WP S owns the restore group, due at the freeze; D20's rule for a frozen set with missing arrays; `run_75cm.sh` is a chain with done-markers, one command per delay tier and a watchdog entry; the bar resumes per episode; "identical after kill and resume" is defined in 4.6.
- **COV-07** accepted, decided by ruling 1. D6, D7, 3.7: the appendage law is a named rule; a leg drive over blocks; a neck or tail over on a turn is frozen as `a1_over` only with that rule (verify test); on a walk or straight move any drive over blocks.
- **COV-08** accepted. D21's ordered fallback; 3.7's table of blocking gates; "not run" records in the bar's schema.
- **COV-09** accepted. Checked `clearance.py:739-758, 1000-1013` and `look.py:867-888` (substeps default 0; cyclic swing runs). 4.5's tests; `screen_move` refuses a window that ends inside a swing; the keys `sets verify` reads are listed in 3.7; S depends on C in section 5.
- **COV-10** accepted. Section 5: two packages at a time with compute caps; the legacy check's record serves the run plan; `Laws` and `PlanCtx` have signatures and the look's solver side lives in `moves_look.py`; the regression's hashes are in a record, not constants; worktree inputs by `restore_inputs.py copy-from`; `env.py` and `panels.g7` are on the K-stride consumer list.
- **COV-11** accepted, option (a). Same as MC-13; `bar.reference_recorder` plays any track kinematically and `bar refcheck` reads every reference with the bar's readings; `readings.reference_readings` is limited to what a file holds; `R0@10ms` readings are checked in the slow bar test.
- **COV-12** accepted. Same as MC-12; the time estimate is "4 solves each way" and the full-pass case is slow.
- **COV-13** accepted. 4.2: episode records are compared value for value but for `wall_s`; the run refuses without arrays or model files and asserts a clean tree before and after.
- **COV-14** accepted. D9: each turn is gated at `(pi/2)/rate`; the 80 degree rule also holds at the slowest clock; episode lengths by 3.8's table; an incomplete judged window is an error.
- **COV-15** accepted. Section 6: one step per delay tier; attribution of the judged delay first with its own worst-case line; the cheap JPS rows are T3.7 and the rest print "not run"; the table is recomputed from `dry-run` into `COST.md` before T3.1.
- **COV-16** accepted as a pre-registered rule, the shared slot kept (open question 6). D12, D13: `fell_before_command` and `command_s` in every episode; reports split falls; attribution cites the walk record when fewer than 5 falls come after the command.
- **COV-17** accepted. Terms are `toe_clearance_own_swing`, `stopped_by_end`, `stop_acceleration_1hz`, `touchdowns_after_command`; `stage1.*` keeps Stage 1's; `join_a` and `stop` are `null` for causal moves and Stage 1's stop reading sits under `reference.stage1_stop` with its command phase labelled; `fn_pre` is replaced by `rule_f` and 3.1 says the pre-command reference always uses gait a's constants.
- **COV-18** accepted. D1 and D11: the no-loss conditions run through the bar's own runner on the `S1` view (stated; the stock `eval` route is dropped); `Delayed` wraps any `panels.Row` with merged sources and k up to 10; `setio.add_events` and `build_track` are shared; the bar's own record functions are listed in 2.3; the worker's exception path is tested.
- **COV-19** accepted. D3: the crouch comes from the spec per body and `sets verify` asserts one per set; D16 is a pre-registered rule and records hold `omega` and `crouch_frac`; the sensitivity control is offered as open question 11.
- **COV-20** accepted. D12 names what round A leaves out; the turn-rate and nudge ladders are T4.6 (last in the order, likely cut).
- **COV-21** accepted. 4.7's list of eleven checks.
