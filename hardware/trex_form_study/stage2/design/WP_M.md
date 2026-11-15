# Work package M: `moves.py`, the generalised builder of starts, changes, stops and finite turns

You are building one module of a robotics simulation study (a robotic T. rex biped; everything is simulation).
You need no other context than this brief and the design it points into:
`/tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/design/ROUND_A_DESIGN.md`
("the design"; read its sections 0, 1 (D4-D8), 2.2, 3.1-3.6, 3.9's two dataclasses, 4.1 before you write code).
Prototypes that already work, to port and not to import: `.../scratchpad/design/proto_moves.py` (`gplan`,
`walk_correction`, `gait_from_spec`, `periodic_ref`, `hip_yaw_use`), `proto_causal.py` (`solve_causal`), and the
reviewer's `.../design/review_mech/r6_taper.py` (the fade of step 5).

## Goal

`designlab/stage2/moves.py`: one builder that (1) reproduces every one of Stage 1's 18 transitions bit for bit
through a schedule passed as data, and (2) builds causal commanded moves (the two-touchdown stop and the finite
90 degree turn) for any `gait.GaitSpec` on any body. It writes no set file: it returns arrays and summaries.

## Files you own (and no others)

- `designlab/stage2/moves.py`
- `tests/test_stage2_moves.py`
- `stage2/sets/legacy_check.json` (made by your CLI; commit it)

Do not edit any other file. In particular: nothing under `designlab/stage1/`, not `designlab/gait.py`,
`turning.py`, `statics.py`, `designlab/stage2/look.py`, no committed record. Do not create `s2io.py`, `setio.py` or
`.gitignore` files (work package R owns them and runs beside you); until R is merged, write your one record with
four lines of your own (`json.dump` to `<path>.tmp`, `os.replace`) and read `RAMP_S = 0.375` from a module
constant of your own that R's `setio.RAMP_S` will later replace by import (leave a comment).

## Setup

Worktree: the absolute path the lead gives you; work only there. Never `cd` the lead's shell into it.
```
cd <your worktree>/hardware/trex_form_study
PY=/home/user/mesozoic-labs/hardware/trex_form_study/.venv/bin/python
export PYTHONPATH=. MUJOCO_GL=disable OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
$PY tools/restore_inputs.py copy-from /home/user/mesozoic-labs/hardware/trex_form_study --group picks-models picks-arrays
git status --porcelain        # must be empty; if link directories show, add them to .git/info/exclude
```
One heavy process at a time (`--procs 1`; another builder and a restore job share the 4 cores); run anything over
a minute through `bash tools/lowprio.sh`. Write logs under a `logs/` directory (ignored). Compute cap: 1.5
core-hours in all.

## API (frozen; ask the lead before changing a name or a signature)

Exactly the listing of the design's section 2.2: `GaitX` (with `in_place`), `Schedule` (with `rule_f`), `Move` (with
`builder`, `args`, `definition()`), `legacy_schedule`, `start`, `change(name, a, b, ramp_strides=1.0)`,
`stop_one_step`, `stop_two_step(name, a)`, `turn(name, a, angle, rate, stride_factor, ramp_s=0.375)`, `stage1`,
`stage1_gait`, `plan`, `generate`, `kinematic`, `arrays`, `summary`, `pick_stride_factor`, `legacy_check`; and
section 3.9's `PlanCtx`, `Laws`, `TODAY` with the dispatch `if move.laws.solver == "look":
importlib.import_module("designlab.stage2.moves_look")` in `generate` (the module does not exist yet; `TODAY`
never reaches it). Module constants: `STRIDE_GRID = (1.00, 0.95, ..., 0.60)`, `TWIST_KINEMATIC = 0.85`,
`TWIST_FINAL = 0.87`, `STEPS_DOWN_MAX = 2`, `CAUSAL_FADE = (0.5, 1.0)`, `CONSISTENCY_LIMIT_M = 2.5e-3`,
`ZMP_HISTORY_LIMIT_M = 4.0e-3`, `CAUSAL_COP_MARGIN_M = 0.010`, `JOINT_AT_COMMAND_RAD = 1e-3`, and `RULES`, a dict
of rule id -> text (`S2A-causal`, `S2A-rule-F`, `S2A-fade`, `S2A-ramp`, `S2A-appendage`, `S2A-stride`,
`S2A-gates` with the sentence "constants set on the 14 Nov prototypes of V0 on 75-M-H-10, before any closed-loop
run").

The module starts and ends with the CLI skeleton of the design's section 2 (thread variables set before numpy; the
`__main__` guard imports `designlab.stage2.moves` by name and calls its `main`). A `Move` holds lambdas: never
pickle it; a pool worker gets `(body, name)` and builds the move itself.

## Algorithms

- `plan`: design 3.1, items 1-6. It is `transitions.plan` (`designlab/stage1/transitions.py:296-426`) line for
  line with the schedule as data. Raise `ValueError` for a causal schedule whose command is not a multiple of 0.5.
- `generate`, non-causal with `TODAY`: design 3.2, first paragraph. Call `transitions.initial_state`, `com_path`,
  `solve_frames`, `derivatives`, `multibody_zmp`, `idyn` themselves, in `transitions.generate`'s order
  (`:638-678`), so the floating-point operations are Stage 1's.
- `generate`, causal: design 3.2 steps 1-6, including the incremental preview form and the fade
  `1 - s((p - c - 0.5) / 0.5)` on the excused transient; record the three histories.
- `GaitX.legacy()`: `transitions.gait_of`'s constructor call argument for argument (`:165-179`); an in-place turn
  has speed 0.0.
- `turn`: design 3.4 (ramps of 0.375 s of reference time: `r = 2 ramp_s / (T_w + T_t)` strides; the `kappa`
  closure; the tail's and neck's amplitudes times `(T(p)/T_w)^2`). `stop_two_step`: design 3.5.
- `summary`: design 3.3 and 3.6. Start from `transitions.summary`'s keys but: no `seconds`; for a causal move
  `join_a = None` and `stop = None` (do not call `compare(g, ref_a, "a")` or `stop_time`), and add `join_command`
  (with `at_command.zmp_mm`), the histories, `cop_margin_single_min {m, frame, foot}`, `range_use`,
  `ik_fail_frames`, `turn {...}`, `reference {...}` (headings at the bar's clock and at the slowest clock, or the
  stop's touchdowns, stopped time, 1 Hz and raw deceleration, margin before the command, and Stage 1's stop
  reading under `stage1_stop` with its command phase labelled), `move {...}`, `definition`, `pass_`.
  For the reference's own readings use small local helpers on the arrays (heading(t), planned `support`,
  `com`); a zero-phase order-2 Butterworth at 1 Hz as `designlab.stage1.metrics.lowpass`.
- `pick_stride_factor`: design D7 and 3.4's last item. Both directions' `kinematic()` per f, descending.
- `legacy_check`: for each body and name, `generate` at full passes into a temporary directory
  (`np.savez(open(tmp, "wb"), **arrays(g))`), sha256 against `stage1/runs/2026-11-08/<body>/transitions.json`;
  the record of design 3.6 (`legacy_check.json`), merged per body, `complete` only when all 18 names ran.

## Tests to write (`tests/test_stage2_moves.py`)

Every test of the design's section 4.1, both tables, with those names. Fast ones in the default suite; slow ones
behind `STAGE2_SLOW=1` (see `tests/test_stage2_look.py` for the idiom); tests that need the Stage 1 arrays skip
without them. Tests that build arrays run them in a subprocess with the three thread variables set to 1. Do not
run the whole suite's slow tests more than once.

## Acceptance (run these, in this order, and report each command's own exit code)

```
$PY -m pytest -p no:cacheprovider tests/test_stage2_moves.py                      # fast: all pass
STAGE2_SLOW=1 $PY -m pytest -p no:cacheprovider tests/test_stage2_moves.py -k "legacy_generate_equals_stage1 or stop_and_turn"   # once
bash tools/lowprio.sh $PY -m designlab.stage2.moves legacy --body 75-M-H-10 --procs 1 --record stage2/sets/legacy_check.json
bash tools/lowprio.sh $PY -m designlab.stage2.moves legacy --body 60-N-T-10 --procs 1 --record stage2/sets/legacy_check.json
$PY -m pytest -p no:cacheprovider tests/test_stage1.py -k "transition" tests/test_stage2_look.py -k "not regenerates"   # unchanged, passing
git status --porcelain            # only your three files
```
Done when: 36 of 36 legacy sha256 are equal and recorded; the scratch build of `stop2_v0.4` and `turn90_pi4_right`
from the V0 0.40 walk on `75-M-H-10` passes every gate of design 3.3-3.5 (expect about: stop consistency under
1 mm, two touchdowns, 1.2 strides; turn `ik_ok`, hip-yaw use about 0.82, join CoM about 0.1 mm and ZMP under
1 mm with the fade; these are scratch numbers, not targets: report what you get); the kinematic pick is 0.80 for
V0. Report any gate that fails with its numbers rather than changing a constant.

## Commit

Commit on your own branch in your worktree (`git -C <worktree> ...`), small commits, messages ending with the
attribution lines your session's system reminder gives, exactly; name no model anywhere else. Do not merge, do
not push unless the lead says so, never rewrite history. Final report: files changed, each acceptance command
with its exit code, the numbers of the scratch stop and turn, anything you could not do.

---

## LEAD ADDENDUM 2, 15 Nov: RULINGS AFTER THE TWO REVIEWS (the fixer follows these)

These settle the points the reviewers handed up. Where a ruling and a reviewer's proposed fix differ, the ruling wins.
The lead writes the matching words into `STAGE2_75_PLAN.md`; you do not edit that file.

**M-R1. The two causal ZMP limits each read their own frames** (mechanics finding 1, bit-identity finding 1). Take
the reviewer's recommended fix, as follows.
- `zmp_consistency` (limit 2.5 mm, unchanged) is the largest |e| over the output frames with phase < command + 1.
- `post_window_zmp` (limit 4.0 mm, unchanged) reads the frames from one stride after the command on, as now.
- Keep the all-frames value in every entry as a recorded field, `zmp_consistency_all_m` (and its history if the
  history is kept per pass), gated by nothing.
- `RULES['S2A-gates']` says exactly: "consistency error over the first stride after the command at most 2.5 mm; ZMP
  error from one stride after it at most 4.0 mm (windows fixed on 15 Nov after the scratch builds of V0, before any
  set was frozen and before any closed-loop episode on a Stage 2 set: STAGE2_75_PLAN.md A7)".
- This is a rule amended after a prescribed-motion result and is labelled so in the plan. Neither constant moves.
- Update `test_causal_fade_window` to the windowed reading, and restructure `test_stop_and_turn_from_v0` so that one
  failed gate cannot hide the other readings (bit-identity finding 1, part 2). Expected under this reading: V0 turn
  2.04 mm / 2.76 mm, V0 stop 1.62 / 0.53. If your build does not reproduce these to the digits the reviewer logged,
  stop and report; do not tune anything.

**M-R2. The acceleration reading** (mechanics finding 2). Take the reviewer's fix, both parts together: second
differences on the frames' own time grid, then interpolation for the filter; the filter is given the walk (or the
stand) before the first frame and after the last as its history; both peaks are read over the move's own frames
only, and `accel_filter` says so. The gate's constant does not move. Expected for V0's stop: low-passed 0.623, raw
1.90 m/s^2.

**M-R3. The 1.0 rad/s ladder turn** (mechanics finding 3). Option (a): no code change. Rule A11's third outcome
already covers it: the turn is built, written with `pass_: false` and its numbers, and not frozen; the set builder
records "no reference within the heading rule" for that rate, and it goes to the owner in the report. The
slowest-clock gate stays on every rate.

**M-R4. Everything else.** Fix bit-identity findings 2 to 7 and mechanics findings 4 and 5 as the reviewers propose.
For bit-identity finding 4, restore `gait.ACCURACY` on exit wherever this module sets it, and say in the docstring
which calls set it. For finding 6, the default suite writes nothing under the study tree (use `tmp_path`).

**M-R5. After your last edit to `moves.py`**, make `stage2/sets/legacy_check.json` again for both picks (about 55 min
of CPU, one heavy process at a time through `tools/lowprio.sh`), and report the tool's own last line. The record must
show 36 of 36 equal.
