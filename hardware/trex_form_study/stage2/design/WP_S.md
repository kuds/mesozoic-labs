# Work package S: `sets.py`, building, verifying and freezing reference sets

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
> - ALSO YOURS (15 Nov): the Jetson body `75-M-H-10-J` is merged on the working branch (merge that branch into your
>   worktree first: `git -C <your worktree> merge --no-edit claude/mesozoic-trex-stage2-refine-4jffre`). Its git-ignored
>   inputs are listed in `stage2/j/j_inputs.json` (30 files: 2 model files, 10 references, 18 transitions) with the
>   helper `stage2/j/j_inputs.py` (the `-X` helper's interface, one BLAS thread, needs no other group). While you add
>   the `sets` group to `tools/restore_inputs.py`, add a group `j` for them in the same way the `x` group is done
>   (`GROUPS`, `tools/README.md`'s table and counts, `tests/test_restore_inputs.py`). The body is registered
>   (`designlab.stage1.BODIES`), so `sets build --body 75-M-H-10-J` must work like any other body; do not build its
>   sets now.


You are building one module of a robotics simulation study (a robotic T. rex biped; everything is simulation).
You need no other context than this brief and the design it points into:
`/tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/design/ROUND_A_DESIGN.md`
("the design"; read its sections 1 (D2, D3, D7, D14, D19, D20, D21), 2.1, 2.3's `sets.py` block, 2.4's third row,
3.3-3.7, 4.3).

Background. A "reference set" is a directory `stage2/sets/<set>/<body>/` in Stage 1's run-directory layout, so
that Stage 1's track and planner code loads it unchanged. Three sets are built now on body `75-M-H-10`: `V0`
(today's walk, a control), `F25F9` (a tuned walk from a pull request, a control) and `F25F9cl` (that walk with a
duty and lift margin: the candidate). Each holds periodic walks at the nominal speed and at 0.30-0.50 m/s, a
start, speed changes, a two-touchdown stop, a one-step stop for comparison, and 90 degree turns in both
directions at four yaw rates. The arrays are git-ignored; the JSON records are committed; a set is "frozen" by a
manifest of sha256, and panels then run only on frozen sets. Your module must work for any set and body: nothing
may be hard-wired to `75-M-H-10` or to these three sets.

Merged before you start: `moves.py` (work package M: builds any move and returns arrays and a summary),
`s2io.py` and `setio.py` (R: atomic writes, locks, `r2()`, `built_by`, names, the entry subsets, `read_inputs`),
`setscreen.py` (C: writes `screen.json`). Read their docstrings; do not edit them.

## Files you own (and no others)

- `designlab/stage2/sets.py`, `tests/test_stage2_sets.py`
- the edits to `tools/restore_inputs.py`, `tools/README.md`, `tests/test_restore_inputs.py` (the `sets` group)

## Setup

Worktree: the absolute path the lead gives you; work only there.
```
cd <your worktree>/hardware/trex_form_study
PY=/home/user/mesozoic-labs/hardware/trex_form_study/.venv/bin/python
export PYTHONPATH=. MUJOCO_GL=disable OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
$PY tools/restore_inputs.py copy-from /home/user/mesozoic-labs/hardware/trex_form_study --group picks-models picks-arrays look
```
One heavy process at a time (`--procs 1`; another builder works beside you); long commands through `bash
tools/lowprio.sh`. Compute cap 1.5 core-hours: build `V0` on `75-M-H-10` once in your worktree; the lead builds
the three sets in the main checkout afterwards. Do not freeze anything in a committed directory: freezing is the
lead's act.

## API (frozen)

The `sets.py` listing of the design's section 2.3, with the CLI skeleton of section 2. `Item` is a small
dataclass `(name, kind, depends: tuple[str, ...], definition: dict)`; kinds `walk`, `start`, `change`,
`stop_two_step`, `stop_one_step`, `turn`.

## What to build

1. **Definitions** (design D3). `v0_spec(body, H, v)`; `tuned_f9_spec(body, H, v)` copied verbatim from
   `stage2/jetson_plem/jetson_plem_study.py:369-390` (do not import that script at run time);
   `SETS = {"V0": ..., "F25F9": ..., "F25F9cl": ...}` with labels (`"new"` for V0 above `envelope.FAST`'s speed
   and for F25F9cl's rungs other than 0.40, 0.45, 0.50 m/s) and F25F9cl's `margin_rule` text.
   `items(set, body)`: six walks, `start`, `nominal_to_v<v>` for each of the five speeds, `stop2_v0.4`,
   `stop1_v0.4`, and `turn90_<rate>_<dir>` for the four rates of `setio.TURN_RATES` and both directions, all
   from `v0.4`. Names only through `setio`.
2. **A walk** (`walk_arrays`, inside one `s2io.r2()` block): `g = look.generate(src, spec)`; `idr =
   gait.inverse_dynamics(M, g)`; `summary = gait.summarize(M, g, idr)`; Stage 1-format arrays = `refs.arrays(src,
   dict(name=..., traj=g, idr=idr, summary=summary, ...))` exactly as `refs.generate` assembles its dict
   (`designlab/stage1/refs.py:113-134`); look-format arrays = `look.arrays(src, g)`; the look entry =
   `look.measure`, `look.price`, `look.flags` as `look.write` computes them (`look.py:1349-1368`) without the wall
   times. Entry: the design's 3.6 (`refs.json`). Gate at build time: `q, qd, qdd` of the two formats
   `array_equal`; for V0 `nominal` the Stage 1-format sha256 equals `stage1/runs/2026-11-08/<body>/refs.json`'s
   and for V0 `nominal` and `v0.4` the look-format sha256 equals `stage2/look/look_refs.json`'s V0 rungs
   (`nominal`, `v040`); a difference stops the build with both hashes. If `qd` or `qdd` differ although both formats were computed
   inside the same `r2()` block, report the largest difference and stop: do not loosen the gate yourself.
3. **A move**: `moves.start / change / stop_two_step / stop_one_step / turn`, `moves.generate`, `moves.arrays`,
   `moves.summary`, inside `s2io.r2()`; the file through `s2io.write_npz`; the entry = the summary + `file`,
   `sha256`, `built_by`. For V0 `start` the sha256 must equal Stage 1's (`transitions.json`).
   Turns: `moves.pick_stride_factor` once per rate (record under `set.json` `stride_rules[rate]`), then build both
   directions at the chosen f; the step-down of D7 (at most twice) if the built reference exceeds 0.87 or lacks
   `ik_ok`; no f -> both directions `available: false` with the numbers.
   Fallbacks for other moves: D21, in that order and no other; record `fallback`.
4. **`build`** (design D19): `s2io.require_one_thread()`; under `s2io.Lock(set_dir / ".lock")`; `s2io.sweep_tmp`;
   merge any leftover `partial/<name>.json` whose file is present with the right sha256; then run the missing
   items in dependency order with a spawned pool of `procs` workers (`multiprocessing.get_context("spawn")`),
   each worker getting `(set, body, name, root)`, building `src` itself (`look.source(body)`), calling
   `build_item`, writing the npz files atomically and `partial/<name>.json`, and returning the name; the parent
   merges each finished item into `refs.json` or `transitions.json` and `set.json` (atomic) and deletes the
   partial **in the same step**. An item is skipped when its entry exists, its `definition` equals the one
   `items()` gives now, and its file's sha256 matches. No wall time in any committed JSON (log it under
   `logs/`). **On a frozen set**: an item listed in the latest manifest is never rebuilt into the JSON; if its
   array is missing, rebuild it in a temporary directory and place it only if its sha256 equals the manifest's
   (else fail loudly, naming both hashes); entries not in the manifest may still be added.
5. **`verify`**: the design's 4.3 items 1-6, reading `screen.json` only through the keys listed at the end of
   3.7. Returns `{ok, failures [...], notes [...]}`; exit 1 on a failure.
6. **`freeze`** (design D20): requires `verify` ok, a complete `screen.json`, and
   `stage2/sets/legacy_check.json` passing for both picks with `moves_sha256` equal to today's `moves.py`;
   appends `{index, at, commit, manifest {name: {kind, sha256, entry_sha256}}, manifest_sha256,
   legacy_check_sha256, mixed_sources, dirty}` to `freezes`; refuses to drop or change any `(name, sha256)` of
   the previous manifest; refuses mixed `built_by` source hashes or a dirty tree unless `--allow-mixed` /
   `--allow-dirty` is given, and then records the flag.
7. **The restore group** (design 2.4, third row): in `tools/restore_inputs.py` add `"sets"` to `GROUPS` (last),
   `NEEDS["sets"] = ("picks-models", "picks-arrays")`, a listing made from every
   `stage2/sets/*/*/set.json`'s latest manifest (each `refs/`, `transitions/` and `look/` npz with its sha256),
   and a restore that runs `python -m designlab.stage2.sets build --set S --body B` per frozen set (which, on a
   frozen set, only places matching arrays). Update `tests/test_restore_inputs.py` (`:96` the group tuple, `:79`
   the file count, which must become a count computed from the manifests for the `sets` part) and
   `tools/README.md`'s group list and counts in the same commit. With no frozen set the group lists nothing and
   every existing test must still pass.

## Tests to write (`tests/test_stage2_sets.py`)

Every test of the design's section 4.3 ("Tests" paragraph). Use a tiny synthetic set (hand-written JSON and
small npz files) for names, keys, freeze, resume, lock and ignore tests so they are fast; mark the ones that need
a built 75 cm set "built" and skip them without it; the two-process-state test is slow (`STAGE2_SLOW=1`). The
kill-point test can inject a failure through an environment variable your builder reads only in tests
(`STAGE2_SETS_FAIL_AT=<point>`), then rerun and compare the files.

## Acceptance (report each command's own exit code)

```
$PY -m pytest -p no:cacheprovider tests/test_stage2_sets.py tests/test_restore_inputs.py
bash tools/lowprio.sh $PY -m designlab.stage2.sets build --set V0 --body 75-M-H-10 --procs 1      # about 40 core-min
bash tools/lowprio.sh $PY -m designlab.stage2.setscreen run --set V0 --body 75-M-H-10 --procs 1
$PY -m designlab.stage2.sets verify --set V0 --body 75-M-H-10
$PY -m designlab.stage2.sets build --set V0 --body 75-M-H-10 --procs 1                            # second run: nothing built, no file rewritten
$PY -m pytest -p no:cacheprovider tests/test_stage2_sets.py tests/test_stage2_setscreen.py        # the "built" tests now run
git status --porcelain            # JSON and Markdown under stage2/sets/V0 only, plus your code; no npz, tmp, lock, partial
```
Done when: V0's `nominal` and `start` hash to Stage 1's and its look-format `nominal` and `v0.4` to the ladder's;
`verify` passes or names exactly which references failed which gate (a turn rate with no stride factor, a rung
over a drive limit and a move that needed a fallback are results, not failures of your code: report them with
their numbers); the second build does nothing; a build killed at each injected point resumes to the same files.
Do not commit the built V0 directory's JSON unless the lead asks (the lead rebuilds and freezes in the main
checkout); do report the stride factor picked for each rate, every turn's hip-yaw use, every `a1_over`, every
fallback and every "not available" item.

## Commit

Commit on your own branch in your worktree, messages ending with the attribution lines your session's system
reminder gives, exactly; name no model anywhere else. Do not merge, do not push unless the lead says so.

---

## LEAD ADDENDUM 2, 15 Nov: RULINGS AFTER THE REVIEWS OF M, B, C AND G (read before you build)

The committed rules were amended on 15 Nov after the builds' independent reviews; `STAGE2_75_PLAN.md`'s last section
lists the amendments and each stands, labelled, in its rule (A7, A11, C3, C5, R4). The rulings as given to the
fixers are in `REVIEW_RULINGS_15NOV.md` beside this brief. What they mean for the sets:

- **Build on the fixed packages.** Merge the working branch and the fixed branches of M, C, B and G before you start,
  and say in your report which commits you built on. If a fixer's branch is not finished, stop and report; do not
  build sets on an unfixed `moves.py`.
- **A7's gates are M's as fixed**: the consistency limit reads the first stride after the command, the 4.0 mm limit
  the frames after it; every entry also holds `zmp_consistency_all_m`. You change no gate and no constant. A move
  that fails a gate is written `pass_: false` with its numbers and is not frozen (A11's third outcome); you do not
  retry it another way.
- **The stride rule also reads A12's 0.10 rad (A11 as amended on 15 Nov). This edit is yours.** In
  `moves.pick_stride_factor` (and `moves.kinematic` where it helps) record, for every `f` tried and both directions,
  the tightest joint margin other than the thigh twist (joint name and rad, against `plant.usable_ranges`), and pick
  the largest `f` on the grid whose kinematic pass has `ik_ok`, uses at most 0.85 of the thigh-twist range on the
  worse leg **and leaves every other gated joint at least 0.10 rad of range**. The built reference must show `ik_ok`,
  at most 0.87 of the twist and A12's 0.10 rad; if not, `f` steps down once, at most twice in all, as before. No
  constant changes and A12's gate in `setscreen.py` is untouched. `set.json`'s `stride_rules[rate]` holds the new
  readings. Add a test that pins the pick's logic on a small synthetic table, and a slow test on V0's quick turn.
  The lead's kinematic scan of V0's quick turn on 75-M-H-10 (scratch, prescribed motion) gave a hip-roll margin of
  0.080, 0.091, 0.101, 0.111, 0.121 rad at `f` 0.80, 0.75, 0.70, 0.65, 0.60, so the expected pick for V0 is 0.70;
  report yours, and do not tune anything to it. After your last edit to `moves.py` make
  `stage2/sets/legacy_check.json` again for both picks (one heavy process at a time through `tools/lowprio.sh`).
  Your reviewers review this edit with the rest of your package.
- **The 1.0 rad/s ladder turn** is expected to fail A10's slowest-clock heading (78.4° on V0). Build it, record it
  with `pass_: false`, and write "no reference within the heading rule" for that rate in `set.json`.
- **`set.json`**: each freeze may carry `change_after_result` (a string with the reason; absent when none), which the
  runner copies into its records. The top-level `crouch_frac` equals every walk entry's own `crouch_frac`; the
  runner refuses a set where they differ.
- **Rule R4**: before task 3's first unit the only closed-loop episodes on a registered set are single check
  episodes (seed 3042, one per condition) on V0 under R0 rows, each listed in
  `stage2/runs/2026-11-15-75cm/dev_episodes.json`. Run none under an R2 row.
- **Nothing is frozen by you.** Build, screen and record the three sets; the freeze is the lead's step at code
  state A, after the legacy check and the regression record are made again on the merged code.

---

## LEAD ADDENDUM 3, 15 Nov, 06:30 UTC: AFTER THE FIXES OF M, G, C AND B (for the builder if it reads this in time, and for the fixer)

M, G, C and B are fixed and merged into the working branch (merge commits `4ef2674`, `2d9adef`, `2b12c13`,
`dd6b372`). What their fixers handed to this package:

- **What the runner needs from a set** (B's fixer): `manifest_sha256 = s2io.sha256_json(manifest)`;
  `freezes[i].change_after_result` (absent when none); every walk entry's `crouch_frac` and `speed_m_s` equal to
  `set.json`'s and to the entry's name; each causal move's `command_phase` matching the runner's leads (2, 2.5 and
  2.5 strides). Run `bar refcheck` on each built set and report its own last line.
- **Key names of the screen** (C's fixer): a walk's `arrays.sha256` compares with `refs.json`'s `look_sha256`, and
  `reference.sha256` with `refs.json`'s `sha256`. `screen.json` is schema 2.
- **One acceleration reading for a move (the lead's ruling).** A12's 1.5 m/s² gate on a start, change or stop reads
  the entry's own value, the one `moves` computes under A7 as amended (differentiated on the frames' own grid, the
  filter given the walk or the stand around the move). In `setscreen.judge_move` read the entry's value where the
  entry has one, keep the screen's own no-history value beside it as `com_forward_lp1hz_peak_screen`, and say so in
  `RULES`. In `readings.reference_readings` take the entry's value or call `moves.forward_acceleration` with the
  history, so that the three modules give one number for one reference. Add a test that the three agree on a built
  stop. These are small edits to fixed packages: list each in your report with its commit so that the lead reads
  them.
- **A red test you will meet**: `test_legacy_check_record_is_current` is red after any edit to `moves.py` until the
  legacy check is made again; make it again after your last edit to `moves.py` (addendum 2).
