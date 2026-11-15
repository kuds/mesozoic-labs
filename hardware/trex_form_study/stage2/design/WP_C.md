# Work package C: `setscreen.py`, the Stage 0 screen of every reference of a set

> **LEAD ADDENDUM, 15 Nov 2026 (read before anything else; it overrides the text below where they differ).**
> The rules are now being committed as `STAGE2_75_PLAN.md` (study top level) and `STAGE2_PLAN.md` section 9. Read the
> committed text, not only the draft: if your worktree lacks `STAGE2_75_PLAN.md`, run
> `git -C <your worktree> merge --no-edit claude/mesozoic-trex-stage2-refine-4jffre` (the working branch), or read
> `/home/user/mesozoic-labs/hardware/trex_form_study/STAGE2_75_PLAN.md`; if it is not there yet, read
> `/home/user/mesozoic-wt/prereg/hardware/trex_form_study/STAGE2_75_PLAN.md`. The lead's rulings on its twelve open
> points are in `/tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/design/PREREG_RULINGS_15NOV.md`.
> What they change for the build:
> - Task 3's budget is 16 core-hours, task 4's 12; cuts are made by unit (one set at one delay within a numbered
>   tier; one record's attribution is a unit); the first unit that does not fit ends the task; `budget.json` holds the
>   units in the order started with estimate, spent-before, start time and used core-hours, and the cut units;
>   `COST.md`'s rates are measured on Stage 1's own committed conditions, never on a reference of the three sets.
> - Before task 3's first unit the ONLY closed-loop episodes on a Stage 2 set are single check episodes on set V0
>   under R0 rows, each listed in `stage2/runs/2026-11-15-75cm/dev_episodes.json`. R2 rows and the delay wrapper are
>   smoke-tested only on Stage 1's own run directory (the read-only S1 view). No episode of F25F9 or F25F9cl and no R2
>   row on V0, F25F9 or F25F9cl in any smoke, test or acceptance command.
> - The omega control (F25F9cl `walk/v0.4` at mu 0.9 under R2/42 at 10 ms and 0 ms with Stage 1's 0.03 H crouch in
>   omega, 40 episodes, seeds 3042-3061, no verdict, not replicated) writes under `controls/omega_s1/` with a
>   `control` field; `omega` and `crouch_frac` are in every record's key.
> - Registered records are the first finished ones; a later record for the same set, row, condition, delay and block
>   carries `supersedes.reason` and prints as "re-run after a result" beside the registered one. A reference added,
>   renamed or re-versioned after any episode is labelled a change after a result.
> - Verdict words: `shown` / `provisional` only at 10 ms and only for reading 2 or 3 as named; at another delay
>   "passes at <ms> ms". A turn frozen with `a1_over` has reading 2 "not read: a1_over" (not a pass). A candidate's
>   verdict always prints with "replicated", "not replicated" or "replication cut by the pre-registered order"; R0
>   rows give no verdict and are not replicated.
> - A turn failing any gate has no reference and goes to the owner; no fallback, no constant changed.
> - `stop/one_step` is read from Stage 1's command and labelled so; `walk/nominal` is `start`, then `nominal` to the
>   end, judged over its last 10 s with peak ground reaction, cadence and stride judged as Stage 1 judges them.


You are building one module of a robotics simulation study (a robotic T. rex biped; everything is simulation).
You need no other context than this brief and the design it points into:
`/tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/design/ROUND_A_DESIGN.md`
("the design"; read its sections 1 (D2, D7, D15), 2.3's `setscreen.py` block, 3.6, 3.7, 4.5). A map of the code
you build on: `.../scratchpad/maps/07_screen_motion.md` sections 1.3, 1.7, 6, 7 (the look ladder's screen,
`screen_rung`, `envelope._ratios`, `clearance.screen`).

Background. A "reference set" is a directory `stage2/sets/<set>/<body>/` in Stage 1's run-directory layout:
periodic walks in `refs/<name>.npz` (Stage 1's reference format) with a twin in `look/<name>.npz` (the look
ladder's format), moves (starts, speed changes, stops, finite turns) in `transitions/<name>.npz` (Stage 1's
transition format), with `refs.json`, `transitions.json` and `set.json` beside them. The plan's rule is that every
reference is priced by the Stage 0 screen (drive ratios, passive neck and tail loads, grip need, sway,
accelerations, clearance gates, joint-range margins) before anything is simulated on it. The look ladder has such
a screen (`designlab/stage2/screen.py`) for periodic walks only, writing under `stage2/look/`. You write the
screen for a set, walks and moves, into the set's own `screen.json`, and you must not change one byte under
`stage2/look/`.

## Files you own (and no others)

- `designlab/stage2/setscreen.py`, `tests/test_stage2_setscreen.py`
- one line of `designlab/stage2/screen.py` (`:516`): `SPEED_LABEL[speed]` -> `SPEED_LABEL.get(speed, speed)`

Do not edit any other file. `designlab/stage2/s2io.py` and `setio.py` (work package R, merged before you start)
give you atomic writes, the lock, `require_one_thread`, `r2()`, `load(set_dir)`, names.

## Setup

Worktree: the absolute path the lead gives you; work only there.
```
cd <your worktree>/hardware/trex_form_study
PY=/home/user/mesozoic-labs/hardware/trex_form_study/.venv/bin/python
export PYTHONPATH=. MUJOCO_GL=disable OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
$PY tools/restore_inputs.py copy-from /home/user/mesozoic-labs/hardware/trex_form_study --group picks-models picks-arrays look
```
The sets do not exist yet (work package S builds them after you). Develop `screen_move` on Stage 1's own
transitions (`stage1/runs/2026-11-08/75-M-H-10/transitions/*.npz` with its `transitions.json`: the same format)
and `screen_walk` on a walk you generate in a temporary directory with `look.generate` / `look.arrays` /
`look.price` / `look.measure` (map 07 section 1.7 a). One heavy process at a time; compute cap 0.5 core-hours.

## API (frozen)

```python
def screen_walk(src, set_dir, name, look_entry) -> dict
def screen_move(src, set_dir, name, entry, walk_block) -> dict
def run(set_name, body, root=setio.ROOT, procs=2, log=print) -> dict     # resumable per item; merges into screen.json
def markdown(doc) -> str
def main(argv=None) -> int      # CLI: python -m designlab.stage2.setscreen {run|md} --set S --body B [--procs 2] [--root DIR]
```
`src` is `look.source(body)`. The module uses the CLI skeleton of the design's section 2 (thread variables before
numpy; the `__main__` guard calls the named module's `main`; `main` calls `s2io.require_one_thread()`).

## What each function does

- `screen_walk`: `screen.screen_rung(src, name, <set name>, look_entry, out_dir=set_dir)` inside `s2io.r2()`, then
  (1) drop the block's `seconds` (log it), (2) add `range` from
  `look_entry["price"]["clearance"]["joint_range_margins"]` (`screen_rung` pops that key at `screen.py:495`): per
  joint `{walk_min, walk_max, usable, margin_min, margin_max}`, the tightest joint, `rule_rad: 0.10`, `ok`;
  (3) add `verdict.blocks_freeze` by the table below. `look_entry` is the entry of `look.write`'s shape that the
  set keeps in `set.json` `look_entries[name]` (its `file` is relative to `set_dir`).
- `screen_move`: the design's 3.7 table of blocks, from the npz arrays (`q, qd, qdd, tau, omega, swing, t, com,
  feet, heading, path`) and the entry:
  - `drives`: `sizing._motor_space(M.names, tau, omega, src["layout"])` -> `envelope._ratios(src["design"], bus,
    ms)`; take `bus` as `look.price` does (read `look.py:891-947`); all eight groups;
  - `passive`: `screen.passive_loads(M, design, tau, omega)`; `over_with_passive`;
  - `friction`: single support per frame with `turning.contact_torques(..., "exact")` and the formulas of
    `turning.inverse_dynamics` (`designlab/turning.py:807-824`); double support `turning.ds_min_mu`; `grips` at
    0.9 and 0.54;
  - `sway`, `accel`: on the move's own frames, no wrap (the look screen's versions wrap a periodic window and
    cannot be used); the 1 Hz forward acceleration with `designlab.stage1.metrics.lowpass` on a uniform 500 Hz
    resample;
  - `clearance`: `look._clearance_gates(src, dict(q=q, phases=<support tuples from swing>, ik_ok=entry["ik_ok"]))`.
    First assert that no foot is in swing on both the first and the last frame (the swing runs of
    `clearance._swing_runs` are cyclic) and refuse the move otherwise. `clearance.screen`'s `substeps` stays 0;
  - `range`: per-joint margins against `plant.usable_ranges(body)`; `hip_yaw_range_use` (the entry's);
  - `vs_walk`, `verdict {within_limits, over, near, a1_over, blocks_freeze}`.
- Which gate blocks a freeze: exactly the design's second table in 3.7 (walk / start, change, stop / turn
  columns). `a1_over` is true when a neck or tail group's cont, peak or speed ratio exceeds 1; on a turn it does
  not block, on a walk or a straight move it does.
- `run`: under `s2io.Lock(set_dir / ".lock")`; one job per entry of `refs.json` and `transitions.json` that is
  available and whose block is missing or whose `arrays.sha256` differs from the entry's; each finished block is
  written to `partial/screen__<name>.json` and merged into `screen.json` (atomic), then the partial is deleted;
  workers are spawned with `(set, body, name)` and build `src` themselves. `screen.json`: `{set, body, rules,
  limits (screen.limits(src)), margins, grips, blocks {name: block}, over [..], a1_over [..], blocks_freeze
  {name: [..]}, provenance: s2io.built_by([...])}`; no wall time anywhere in it.
- `markdown(doc)`: one table of every reference (governing ratio, worst leg ratio, neck and tail, CoP margin,
  grip need at both grips, mid-swing clearance, tightest joint, twist use for turns, verdict), then the list of
  what blocks a freeze and the `a1_over` turns, each with its numbers. Units with every number; no adjectives.

## Tests to write (`tests/test_stage2_setscreen.py`)

Every item of the design's section 4.5. The "built" test (V0's `nominal` and `v0.4` blocks equal the committed
`stage2/look/screen.json` rung blocks on every key but `speed`, `variant`, `seconds`, `arrays.file` and the added
`range`) needs the built V0 set: write it now, skipping when `stage2/sets/V0/75-M-H-10/set.json` is absent, and
check the same equality in your own acceptance on walks you generate in a temporary directory from
`sizing.NOMINAL` and `look.speed_spec(H)` (their look-format files must hash to the ladder's V0 rungs, sha256 in
`stage2/look/look_refs.json`).

## Acceptance (report each command's own exit code)

```
$PY -m pytest -p no:cacheprovider tests/test_stage2_setscreen.py
$PY -m pytest -p no:cacheprovider tests/test_stage2_screen.py tests/test_stage2_look.py tests/test_stage2_motion.py -k "not regenerates"   # unchanged, passing
git diff --stat stage2/look            # empty
git status --porcelain                 # only your files
```
Done when: `screen_move` reproduces the recorded single-support CoP margin and friction ratio of Stage 1's `start`
and `enter_walk_0.50_left` on `75-M-H-10`; the temporary-directory walk blocks equal the committed ladder's; the
positive controls fail as designed; a block exists for all 18 Stage 1 transitions of `75-M-H-10` (a smoke of the
whole path: report the governing ratio and verdict of each, as scratch numbers). Report the wall time per block.

## Commit

Commit on your own branch in your worktree, messages ending with the attribution lines your session's system
reminder gives, exactly; name no model anywhere else. Do not merge, do not push unless the lead says so.
