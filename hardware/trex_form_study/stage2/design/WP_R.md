# Work package R: the bar's readings and the shared I/O (`readings.py`, `s2io.py`, `setio.py`)

You are building three small modules of a robotics simulation study (a robotic T. rex biped; everything is
simulation). You need no other context than this brief and the design it points into:
`/tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/design/ROUND_A_DESIGN.md`
("the design"; read its sections 1 (D2, D9, D10, D14, D19, D20), 2 (the opening and 2.1, 2.3's `readings.py`
block), 3.6, 3.8, 4.3 (the `setio` tests), 4.4).

## Goal

1. `readings.py`: the bar's readings as pure functions of arrays. `bar.py` (another package) will call them on a
   live recorder; tests call them on synthetic arrays. Two of these modules are hashed into every bar record's
   key (`readings.py`, `setio.py`), so they must be small, complete and stable before code state A.
2. `s2io.py`: thread check, accuracy context, canonical JSON and sha256, atomic writes, locks, provenance.
3. `setio.py`: names, reading a reference set, the entry subsets a key hashes, command events, the track of a plan.

## Files you own (and no others)

- `designlab/stage2/readings.py`, `designlab/stage2/s2io.py`, `designlab/stage2/setio.py`
- `tests/test_stage2_readings.py`, `tests/test_stage2_s2io.py`, `tests/test_stage2_setio.py`
- `stage2/sets/.gitignore` and `stage2/runs/.gitignore`, each exactly: `*.npz`, `*.tmp`, `.lock`, `partial/`,
  `traces/`, `logs/`, `done/` (one per line)

Do not edit any other file: nothing under `designlab/stage1/`, no committed record.

## Setup

Worktree: the absolute path the lead gives you; work only there.
```
cd <your worktree>/hardware/trex_form_study
PY=/home/user/mesozoic-labs/hardware/trex_form_study/.venv/bin/python
export PYTHONPATH=. MUJOCO_GL=disable OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
$PY tools/restore_inputs.py copy-from /home/user/mesozoic-labs/hardware/trex_form_study --group picks-models picks-arrays
```
Nothing here needs more than seconds of compute.

## API (frozen; ask the lead before changing a name or a signature)

`s2io.py` and `setio.py`: exactly the listing of the design's section 2.1. Notes:
- `s2io.require_one_thread()` raises `S2Error` unless each of `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`,
  `MKL_NUM_THREADS` is exactly `"1"`.
- `s2io.r2()` is a context manager: `gait.set_accuracy("r2")`, assert `gait.ACCURACY == "r2"` on entry and again on
  exit (raise `S2Error` if something inside changed it), then restore the previous level.
- `s2io.write_npz` must write through an open file object (`np.savez(fh, **arrays)`): `np.savez(path)` appends
  `.npz` to a temporary name. Bytes written this way equal `np.savez(path)`'s (checked by a reviewer on a Stage 1
  reference); test it.
- `s2io.Lock`: `os.open(path, O_CREAT | O_EXCL | O_WRONLY)`, JSON `{pid, host, started_at}`; if the file exists, the
  host is this one and the pid is not alive, take it over and log; otherwise raise `S2Error` naming the holder.
- `s2io.built_by(stage2_modules, own_outputs=())`: `git rev-parse HEAD`, `git status --porcelain
  --untracked-files=all` (paths under `own_outputs` left out of `dirty_paths` and of `dirty`),
  `panels.panel_sources()["sha256"]`, sha256 of each named module file, `panels.library_versions()`, the thread
  count, `gait.ACCURACY`, a UTC time.
- `setio` imports `designlab.stage1.control` and `tune` and `s2io` only; never `sets.py`, `moves.py`, `bar.py`.
- `setio.entry_subset(kind, entry)`: the keys of `WALK_SUBSET` or `MOVE_SUBSET` that the entry has (a tuple
  names a nested key); `entry_sha256` = `s2io.sha256_json` of it. A label, a rule text, `built_by`, `look_sha256`
  must not change it.
- `setio.read_inputs(set_dir, files, require_frozen=True)`: `files` are relative paths (`refs/v0.4.npz`,
  `transitions/start.npz`); each file's sha256 must equal its entry's and the latest freeze's manifest (`set.json`
  `freezes[-1].manifest[name].sha256` and `.entry_sha256`); raise `tune.InputError` with the differences. Return
  the two values of section 2.1 (the second in `tune.read_inputs`' shape so that `panels.Planner`, `Context` and
  `tune.build_spec` take it).
- `setio.stage1_view(run_dir, body)`: the same two values for `stage1/runs/2026-11-08/<body>` (no freeze; sha256
  from the JSON entries; `entry_sha256` of the same subsets).
- `setio.commands(plan, tr_doc)`: for each `("transition", name)` whose entry has `move.causal`: `{index, name,
  kind: entry["move"]["kind"] ("stop" | "turn"), command_phase, speed_m_s: gaits.a.speed, period_s:
  gaits.a.period}`. `add_events(track, plan, tr_doc)` appends `dict(kind, name, t_ref=<that segment's t0>,
  command_phase, speed_m_s, period_s)` to `track.events` for each and returns the list. `control.run` reads only
  events of kind `"stop"` (`designlab/stage1/control.py:2325`), in the same shape as its own (`:1000-1001`).
- `setio.build_track(p, set_dir, plan, omega, tr_doc)` = `control.build_track(p, set_dir, plan, omega)` then
  `add_events`.

`readings.py` (numpy and `scipy.signal` only; no MuJoCo, no Stage 1 import): the listing of the design's section
2.3 and the definitions of section 3.8:
- `toe_clearance_own_swing(t_u, low, contact, dt, ref_swing, ta, tb, air_gap_max_s, mid=(0.25, 0.75))`: 3.8 (a),
  every clause: air runs; swing if any spanned row has `ref_swing == i` or `D >= air_gap_max_s`, else air gap;
  interrupted reference swings; the reading with `low` interpolated at the two edges of the middle half plus the
  rows strictly between; returns `{toe_clearance_own_swing_m, own_swings [2], ref_swings [2], interrupted_swings,
  air_gaps, air_gap_longest_s, air_gap_max_s, swings [[foot, t_lo, t_td, min_m, class]], count_ok}`. It does not
  apply the 0.04 H threshold (the caller does). `t_u` are row times; `contact` is bool `[steps, 2]`; `low` is
  `[rows, 2]`; `ref_swing` is `[rows]` with 0, 1 or -1.
- `line_reading`: 3.8 (b). `turn_reading`: 3.8 (c), with `at_s` an argument. `stop_reading` and `touchdowns`:
  3.8 (e), including the touchdown table and the last touchdown before the command.
- `drive_ratios(force_sq, sustained, actuators, leg_mask, k0, k1)`: per drive `sqrt(mean(force_sq[k0:k1], 0)) /
  sustained`; returns `{ratios {actuator: r}, worst [actuator, r], worst_leg [..], worst_a1 [..]}`.
- `reference_heading_at(arrays, tau_s, rate)`: the heading of a transition-format file (`t`, `heading`) at its own
  time `rate x tau_s`, less its first heading, in degrees.
- `reference_readings(arrays, kind, walk=None)`: what a file alone can say: for `"stop"`: touchdowns after frame 0
  from the planned `support`, the time from which the model CoM's forward speed stays under 5 % of the walk's,
  1 Hz and raw forward deceleration peaks; for `"turn"`: headings at the bar's clock, at the slowest clock and on
  the reference's clock, time to 80 deg; for any move the 1 Hz forward acceleration peak. No toe clearance (a
  file holds no toe points; the set's screen has it).
- `lowpass(x, fs, hz=1.0, order=2)`: zero-phase Butterworth identical to `designlab.stage1.metrics.lowpass`.

Each CLI-less module needs no `__main__` guard.

## Tests to write

- `tests/test_stage2_readings.py`: every row of the design's section 4.4 table, with those names and numbers.
- `tests/test_stage2_s2io.py`: the thread check (subprocess with `OMP_NUM_THREADS=4` is refused; unset is
  refused); `r2()` restores the level and catches a change inside; `write_npz` bytes equal `np.savez(path)`'s and
  leave no temporary file; `write_json` is atomic (a pre-existing file survives a simulated failure before the
  replace); the lock refuses a live holder and takes over a dead pid; `canonical` is stable under key order;
  `portable` makes study paths relative; `built_by` leaves `own_outputs` out of `dirty_paths`.
- `tests/test_stage2_setio.py`: every name round-trips exactly (`parse_name(walk_name(0.35))`, all four turn
  rates, both directions); `turn_name("pi4", "left")` maps to `math.pi / 4` exactly; no produced name matches
  `\d\.\d0\b` or holds `0.785`; `entry_sha256` ignores labels and provenance and follows `period_s` and
  `move.command_phase`; `stage1_view` on `stage1/runs/2026-11-08/75-M-H-10` (skip without arrays) gives docs that
  `panels.Planner` accepts; `read_inputs` on a tiny synthetic frozen set refuses a changed file, a changed
  subset and an unfrozen set; `add_events` on a synthetic plan appends one event per causal move with the
  segment's `t0`; `git check-ignore` reports `stage2/sets/x/y/refs/a.npz.tmp`, `refs.json.tmp`, `partial/a.json`,
  `.lock`, `stage2/runs/r/x/traces/a.npz`, `logs/a.log`, `done/a` as ignored.

## Acceptance

```
$PY -m pytest -p no:cacheprovider tests/test_stage2_readings.py tests/test_stage2_s2io.py tests/test_stage2_setio.py   # all pass, under a minute
$PY -c "import designlab.stage2.readings as r, sys; assert 'mujoco' not in sys.modules"
git status --porcelain            # only your eight files
```
Done when the three test files pass, `readings` imports without MuJoCo, and the two `.gitignore` files are in
place. Report any definition in the design's section 3.8 you found ambiguous, with the reading you chose: these
functions are pre-registered rules, so say so rather than decide silently.

## Commit

Commit on your own branch in your worktree, messages ending with the attribution lines your session's system
reminder gives, exactly; name no model anywhere else. Do not merge, do not push unless the lead says so. Final
report: files changed, each acceptance command with its exit code, the ambiguities above.
