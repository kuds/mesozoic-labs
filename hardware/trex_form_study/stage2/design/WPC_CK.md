# Work package CK: the look measures: `lookread.py`

You are building one part of a robotics simulation study (a robotic T. rex biped; everything is simulation). You
need no other context than this brief and the two texts it points into:
- the design: `/tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/design/ROUND_C_DESIGN.md` (read its header, the decisions and sections named under "Read first" below);
- the rules fixed before the runs: `/tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/design/PREREG_TRACKB_draft.md` (rule ids such as "K3" or "W2" refer to it).
Round A's design, where a round-A API is named: `/tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/design/ROUND_A_DESIGN.md`.

**Read first:** design D17, section 2.2 (`lookread`), 3.5 (series, window, four numbers, terms, too few steps), 3.7 (G9-ref (b), (c): what the gate will ask of your functions); rules U1-U5. Also `designlab/stage2/look.py` (`_fit_phase`, the hip-midpoint and head-pitch code near lines 546, 600-646) and `designlab/stage2/motion.py` (`BANDS`).

## Goal

The four look measures of the plan's reading 1 (bob, head pitch in the world, tail phase, toe-off) as pure functions of 50 Hz poses and 500 Hz foot loads, the comparison with a reference's values against the bands, and the reference's values by kinematic playback of a track. No bar module is edited; the wrapper that calls you inside an episode is another package's (`lookbar.py`, package CW).

## Files you own (and no others)

- `designlab/stage2/lookread.py`
- `tests/test_stage2_lookread.py`

Do not edit any other file: nothing under `designlab/stage1/`; no round-A module (`designlab/stage2/bar.py`,
`readings.py`, `setio.py`, `s2io.py`, `moves.py`, `sets.py`, `setscreen.py`, `barreport.py`, `moves_look.py`); no
other package's file; no committed record; nothing at all inside `/home/user/mesozoic-labs` (the main checkout:
you work in your own worktree). If you believe an edit elsewhere is unavoidable, stop and tell the lead: the design
claims none is needed (its section 2.3).

## Setup

Worktree: the absolute path the lead gives you; work only there.
```
cd <your worktree>/hardware/trex_form_study
PY=/home/user/mesozoic-labs/hardware/trex_form_study/.venv/bin/python
export PYTHONPATH=. MUJOCO_GL=disable OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONHASHSEED=0
$PY tools/restore_inputs.py copy-from /home/user/mesozoic-labs/hardware/trex_form_study --group picks-models picks-arrays look
```
Versions: Python 3.13.18, torch 2.15.0+cpu, stable-baselines3 2.9.2, gymnasium 1.4.0, mujoco 3.10.0, numpy 2.5.4.
Needs from other packages: `playback` needs round A's `bar._Played` (branch `wt/wp-b`): import `bar` inside the function; its test skips until `designlab.stage2.bar` imports on your branch. Everything else: wave 1, start now.

## Machine-sharing rules (4 cores, shared with other builders and with running panels)

- One heavy process at a time, always `nice -n 15` (or through `tools/lowprio.sh`); at most 2 worker processes in
  anything you start; nothing longer than about 10 minutes without telling the lead first.
- Never run the full test suite (`pytest tests/` or `pytest` without a file): run your own test files only, slow
  ones (`STAGE2_SLOW=1`) one file at a time.
- Every job you write must be resumable and idempotent (the container restarts without warning) and bounded in
  processes and memory.
- **Episodes (design D23, rule X1):** until the lead tells you that task 3's first unit of the 75 cm round has
  run, no R2-mode, policy or training episode on the sets V0, F25F9 or F25F9cl; no closed-loop episode on a look
  set (V6, V6noF5) before its gate G9-ref has passed. Use Stage 1's own run directory through the read-only S1
  view (`setio.stage1_view`, `stage1/runs/2026-11-08/<body>`) or a synthetic set in a temporary directory.

## Engineering rules

New code only under `designlab/stage2/` (and the paths you own). Records are JSON with their rule texts,
provenance (commit, dirty with your own outputs excluded, versions) and sha256, written atomically (round A's
`s2io.write_json`, `sha256_json`, `git_state` once `s2io` is on your branch). Names are formatted with `%g`.
`git check-ignore` must report every path your code writes that is not a committed record. Tests write only under
`tmp_path`; a test that needs arrays, models or torch skips without them. Fast tests in your test file(s); slow
ones behind `STAGE2_SLOW=1`. Every CLI module follows round A's skeleton (thread variables before numpy; `main()`
of the importable copy) and, where the design's 2.1 says so, starts with `tbenv.assign_gl(...)` before any other
import.

## API (frozen)

```python
BANDS = motion.BANDS          # bob_rel 0.30, head_world_pitch_rms_deg 1.5, tail_phase_cycles 0.1, toe_off_rel 0.70 (read the dict's own keys)
MIN_STEPS = 4
def series(p, poses, loads, dt, every) -> dict
def reading(ser: dict, ta: float, tb: float) -> dict
def compare(own: dict, ref: dict) -> dict
def playback(p, track, run_dir, ta_ref: float, tb_ref: float) -> dict
def window(b, rec) -> tuple[float, float] | None
def sources() -> dict
```
- `series(p, poses, loads, dt, every)`: `p` a `plant.Plant`; `poses [n, nq]` full `qpos` every `every` physics
  steps (50 Hz: `every * dt = 0.02`); `loads [steps, 2]` the feet's vertical loads at every physics step. By
  `mujoco.mj_kinematics` on a private `MjData` of `p.m` per pose: `t50`; `hip_mid_z` (the mean height of the two
  hip joints' anchors, as `look.py` computes it); `head_pitch_world` (the head body's pitch in the world about
  home, `look.arrays`' definition); `foot_pitch [n, 2]` (each foot's pitch about its own lateral axis, toe down
  positive: the sign of the look files' `foot_pitch_solved`); `tail` (the tail-yaw joint angle). Contact by
  `readings.contact_flags` when round A's `readings` is on your branch (the 2 % load rule with 20 ms debounce;
  until then implement the same rule locally behind one function and say so), and from it `lift_offs` and
  `touchdowns` per foot (times). Joints and bodies are found by name, never by index (`-X` bodies have 16 hinges).
- `reading(ser, ta, tb)` (design 3.5 "Four numbers"): `bob_pp_m` (mean over the steps wholly inside the window, a
  step from one touchdown of either foot to the next, of max less min of `hip_mid_z`); `head_world_pitch_rms_deg`
  (RMS about the mean); `tail {phase_cycles, amp_rad}` (least squares `A sin(2 pi phi + psi) + c` with `phi` the
  own phase: 0 at each left lift-off, linear in time between; `phase_cycles = psi / 2 pi`); `toe_off_rad` (mean
  over the lift-offs in the window of the lifting foot's pitch at the lift-off instant, interpolated between
  poses); `steps`, `lift_offs` (counts), `window`.
- `compare(own, ref)`: terms `bob` (`|own / ref - 1| <= 0.30`), `head` (`own <= 1.5 deg`), `tail` (wrapped
  difference of the phases `<= 0.1` cycle; not read, passes, when `ref.tail.amp_rad < 0.02`), `toe_off`
  (`own >= 0.70 x ref`; not read when `ref.toe_off_rad < 0.02`); `not_read` (list); **`too_few_steps`**: when
  `ref.steps >= 4 and ref.lift_offs >= 4` and (`own.steps < 4` or `own.lift_offs < 4`): `bob`, `tail`, `toe_off`
  are False, `head` is still read, `too_few_steps True`, no exception. When the reference itself has fewer: raise
  `LookError` (a definition fault). `ok` = all four.
- `playback(p, track, run_dir, ta_ref, tb_ref)`: poses `bar._Played(track, run_dir).q(t_ref)` at reference times
  `ta_ref, ta_ref + 0.02 x rate, ...` (rate 1.05: the nominal commanded clock, so the result is on simulated
  time), loads from `_Played.loads(t_ref, weight)` at every physics step, then `series` and `reading`. `_Played`
  refuses a periodic turning gait: let its `ValueError` pass through as `LookError("not played: ...")`.
- `window(b, rec)` (rule U2): for a bar walk item (`b.cond.judge == "bar_walk"`) the last 10 s of the planned
  episode; for a turn item (`"bar_turn"`) `[t_c + at_s, t_c + at_s + 3 s]` with `t_c = rec.sim_time(b.command["ref_s"])`
  and `at_s` the condition's judged time (2.0 s for the quick turn; read it from the condition as `bar.py` does);
  else `None`. Uses only attributes; do not import `bar` at module level.
- `sources()`: `{path: sha256}` of this file.

## Tests to write (`tests/test_stage2_lookread.py`)

Synthetic series (no MuJoCo) for `reading` and `compare`: each band on both sides of its limit (bob ratio 0.69,
0.71, 1.29, 1.31; head 1.49 and 1.51 deg; tail phase difference 0.09 and 0.11 cycle, also across the wrap at 0.5;
toe-off 0.69 and 0.71 of the reference); segmentation by the robot's own touchdowns ignores a 10 ms shift of the
whole motion (same numbers); "not read" under 0.02 rad for tail amplitude and toe-off, listed and passing; **too
few own steps (3) with a reference of 6: bob, tail and toe-off fail, `too_few_steps` True, no exception**; a
reference with 3 steps raises `LookError`. With MuJoCo and the look ladder's arrays (skip without): `series` on a
kinematic walk built from a look-ladder rung's own poses reproduces `stage2/look/metrics.json`'s numbers for that
rung within G9-ref (b)'s limits (bob 5 %, head 0.1 deg, tail 0.01 cycle, toe-off 0.02 rad): if it does not, stop
and report the differences and the two definitions: rule Z2 sends this back to the lead. Degraded copies of that
playback (pelvis oscillation x 0.5; a 3 degree neck sinusoid; the tail delayed 0.2 cycle; foot pitch zeroed) each
fail exactly their term. A 16-joint (`-X`) plant builds its series when its model is restored (skip otherwise).

## Acceptance

```
nice -n 15 $PY -m pytest -p no:cacheprovider tests/test_stage2_lookread.py
git status --porcelain            # only your two files
```
In your report: the measures of each look-ladder rung you could read against `metrics.json` (a small table).

## Commit

Commit on your own branch in your worktree, messages ending with the attribution lines your session's system
reminder gives, exactly; name no model anywhere else. Do not merge, do not push unless the lead says so. Final
report: files changed, each acceptance command with its exit code, every place where you read the design or a rule
in one of two possible ways (say which you chose: these are rules fixed before the runs, so say so rather than
decide silently), and anything the design asked for that you could not build.
