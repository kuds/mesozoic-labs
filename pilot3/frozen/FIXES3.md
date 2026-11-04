# Walking pilot 3, FIX3 stage: what each finding of the third code review became

2026-11-03 (UTC), 21:40 to about 22:15. Stage folder: SP/work/pilot3/fix3 ("X"). "SP" is the session scratchpad,
"FC" is SP/work/pilot3/frozen_candidate and "T" is SP/work/pilot3/tools. This file is also copied byte for byte to
FC/FIXES3.md, which the criteria name.

Labels: **[measured: source]** means read or computed from the named file. **[inferred: basis]** means reasoning.
The literal commands and their UTC times are in X/CMDLOG.txt.

## Summary

- **The review had no blocking finding.** It had eight minor ones (M1-M8). Every one that was clearly right is
  applied; the rest are applied in the text or left with a stated reason (table below).
- **M1, the one that could flip Q4, is fixed in the habit tool, tested and stated in the text.** The seed mode now
  reads only the very ledger file of the judged run copy (by resolved path), and refuses, writing nothing and before
  the trace, a ledger whose window does not cover the deciding checkpoint. The review's stale copy is refused
  [measured: T/out/fix3_final/habit_refuse_stale_window.txt, stale_ledger_fixed.txt].
- **The test gaps (M2, M3) are closed.** Every mutant the review saw survive is now killed by the tests, except one
  that fakes the seeding record, which no test can catch [measured: X/out/mutation3_run.log.txt,
  mutation3_run_rerun.log.txt].
- **The branch is unchanged** (head 3ae1c95967547ada569ad71aebaa88999afa02cd, tree clean). No commit, push or git
  config change was made.
- **Re-pinned.** T/SHA256SUMS.txt is rewritten after the FIX3 final run; `sha256sum -c` gives 73/73 OK
  [measured: X/out/tools_sha256_check_after.txt]. The FIX2 list is kept as X/before/tools/SHA256SUMS.txt.
- **New criteria sha256: `6daef03df8094df222a9e692485f2d5e40c218270c9573e736ca31d24b3ecbc7`** (965 lines; the input
  was 901 lines, `2e1dc715...`).

## The findings

| # | Finding | Disposition |
|---|---|---|
| M1 | Habit seed mode: the run's own ledger was checked by folder names only, and nothing checked that the window covers the deciding checkpoint | **FIXED in the tool, the tests and the text.** See note 1. |
| M2 | `pilot3_vg.py`: nine mutants survived the tests (factor 4, SE, seed base, episodes per env, both bootstrap branches, the seeding, two `--cell` refusals) | **FIXED (tests, plus a refactor so the tests reach the code).** See note 2. |
| M3 | Habit: mutant H1c (the `eval_episodes.csv` file-name clause removed) survived | **FIXED.** A test refuses `RUN/03_locomotion/other.csv` (and `RUN/eval_episodes.csv`). See note 1. |
| M4 | `pilot3_noise.py`: a damaged input (a cut `gate_progress.npz`) crashed the whole readout | **FIXED.** See note 3. |
| M5 | `pilot3_vg.py` text and documentation: (a) the n/a rule, (b) G's reward weight, (c) the pooled mean V and mean G columns | **(a) and (c) FIXED in the text and the tool's docstring; (b) FIXED in the text, the docstring and the smoke's check.** See note 4. |
| M6 | Scorer: (1) the fallback to the newest zip by file name for a binding without the intact list; (2) `habit_status` ignored a record's `validation_only` | **(2) FIXED in the scorer, its test and the text. (1) FIXED in the text only, code kept.** See note 5. |
| M7 | `--json` overwrites an existing file; `vg --cell` does not check the branch prefix | **The overwrite is FIXED in all three report tools** (`pilot3_vg.py`, `pilot3_noise.py` and also `pilot3_habit.py`). **The branch check is NOT ADDED.** See note 6. |
| M8 | `tests/e2e_parent_cell.py` is pinned but its sha256 was not cited | **FIXED:** the criteria now cite `be256afc...` (the file is unchanged). |

### Note 1: M1 and M3, the habit tool's ledger

What changed in `pilot3_habit.py` (sha256 `eab5f398...` -> `56f1c78d...`):

- `seed_problems` now requires `Path(ledger).resolve() == Path(binding["run"]).resolve() / "03_locomotion" /
  "eval_episodes.csv"`. This one comparison covers the run copy, the stage folder and the file name, so it replaces
  the three name clauses (M3's H1c clause included). A binding with no run is refused as before.
- New `window_problems(lr, deciding_steps, grid)`: after the ledger is read through the deciding steps, and before
  the trace, the seed mode refuses, writing nothing, when the window has no evaluation, when its first evaluation is
  not 131,072, or when its last is 131,072 or more short of the deciding steps.
- New `window_grid`: the grid is the frozen 131,072 (`EVAL_EVERY`). Only a validation-only cell read with
  `--validation` uses its run's recorded `curriculum_cadence.eval_freq`, so the smoke's cell (32,768) still reads as
  a check.
- A missing ledger column is not refused by the window check, because it already reads NOT RUN, as before.
- The record gains `window_check` (grid, deciding steps, first, last, problems).
- `ledger_reading` and `describe_ledger` are unchanged, so the PC ledger validation is byte-identical [measured:
  T/out/fix3_final/rcs.txt].

Why refuse rather than write a NOT RUN record [inferred]:

- A stale or partial copy is an input fault, like the other `seed_problems` refusals. It is fixed by copying the
  judged run's ledger again; it is not a property of the run.
- With no record, the scorer already reads the seed NOT RUN ("no habit record"), so Q4 reads UNREAD and never YES.
- A NOT RUN record would still let a habit found in the stale part read HABIT ("a habit found is not undone"). A copy
  taken before a RESUME can hold rows of the lost segment that the final ledger drops, so such a habit could come from
  a discarded policy.
- The cost: a run whose real ledger genuinely stops short could never be read, even for a PARK in its trace. The seed
  then reads NOT RUN, which the text's "NOT RUN when any could not be read" allows.

Why the coverage rule holds for every legal run [inferred]: one evaluation falls every 131,072 steps of a learn()
call, so any run that trained through its deciding checkpoint has an evaluation less than 131,072 short of it. This
holds for an unbroken run (8,000,000 - 7,995,392 = 4,608), a collapse-stopped one (for example 5,500,000 -
5,373,952 = 126,048) and a resumed one (for example a resume at 2,500,000, last evaluation 7,873,952, 126,048
short). A RESUME needs 2,000,000 or more, so the first evaluation, 131,072, is never lost to one.

Tests (`tests/test_pilot3_habit.py`, 34 -> 45, sha256 `a9169dee...` -> `a19fcd12...`):

- The FIX2 seed tests now build bindings whose run is a real path under `tmp_path`, because the path is compared.
- New tests cover:
  - another file in the run's stage folder (M3);
  - a same-named copy elsewhere, and the judged copy named through a symlink (accepted);
  - full, collapse-stopped and 2,000,000 windows (accepted);
  - the review's cut at 3,800,000 (refused);
  - the tail boundary (131,071 short accepted, 131,072 refused);
  - a copy without its first evaluation, and an empty one;
  - a resumed grid (accepted) and a missing column (left to NOT RUN);
  - the grid rule, with the smoke's 32,768 grid accepted under `--validation` only;
  - no mode overwrites an existing `--json`.
- Two tests run `main`'s seed mode end to end, with stand-ins for the tree's location, git and the trace, so no
  simulation and no git command runs. A stale judged copy is refused before the trace and nothing is written; a
  covering window writes the record with its `window_check`.

On real files [measured]:

- On FIX2's e2e cell, the seed mode reads NO HABIT with 61 evaluations and 20 of 20 full-horizon trace episodes. Its
  reading, ledger reading, trace and deciding checkpoint equal the e2e's record (T/e2e/habit_s42.json), and only
  `window_check` and `tool_sha256` are new [T/out/fix3_final/habit_seed_e2e_cell.*, compare_with_fix2.txt].
- A stand-in cell whose judged copy holds the e2e ledger cut at 3,800,000 is refused, exit 2, nothing written: "the
  ledger's last evaluation ... is 3,670,016, 4,329,984 steps short".
- The same stale file named against the real cell is refused as not the judged run's own ledger
  [habit_refuse_*.txt].
- The review's probe re-run on the fixed functions (X/bin/stale_ledger_fixed.py, the PC round's OFF s42 ledger)
  gives:
  - the full ledger reads HABIT;
  - the stale same-named copy elsewhere is refused by path;
  - the stale copy at the judged path is refused by the window check;
  - a copy without its first evaluation is refused;
  - `other.csv` is refused;
  - the smoke's validation cell with its run's real ledger is read under `--validation` (grid 32,768, 2 evaluations,
    NO HABIT with a quiet trace) and refused without it [T/out/fix3_final/stale_ledger_fixed.txt].

Text changes:

- Run handling: the habit tool's line.
- "Habit", Window: a new "Coverage (FIX3)" sentence.
- The NOT RUN list now includes "no record (the tool refused the ledger, or a record is validation-only)".
- The tool sentence at the end of "Habit".
- The pilot3_habit.py entry under "How it is judged", with the new sha256s.

### Note 2: M2, the V - G tool's tests

The code the pinned text describes was already right (the review's byte-identical re-run). FIX3 makes the untested
parts reachable by the tests without changing what they compute:

- `episode_rows(eps, value_fn, gamma)` is measure()'s own loop, moved out unchanged (V, the time-limit bootstrap
  after a truncation only, G and the loop-value check).
- `seed_all(vn, seed)` is build()'s three seeding lines. It returns the seeds as set, and the record gains `seeding`.
- `cell_problems(binding, head, dirty)` is main()'s three `--cell` refusals, unchanged, in the same order.
- `arg_parser()` is main()'s parser, so its defaults can be tested.
- An unused `import torch` left in build() was removed (ruff F401).

`pilot3_vg.py`: `b633fbd1...` -> `a58d5f4f...`.

New tests (`tests/test_pilot3_vg.py`, 25 -> 31, `c13b3e12...` -> `ead00c7b...`):

- the frozen settings: 6271, 8, 4, 4.0, 0.995, 1,000 and 1e-4, and the CLI's and measure()'s defaults;
- the SE: paired differences m +/- 1 over 4 episodes give SE 1/sqrt(3) exactly. At 3.5 SE nothing is flagged at the
  default 4 but the bucket is flagged at 3; at 4.5 SE it is flagged;
- G through `episode_rows`: a truncated episode adds gamma^(T-k) V(terminal observation). A terminated one adds
  nothing, although it carries a terminal observation as SB3 hands one over. A truncation without one is a problem;
- `seed_all` on a stand-in VecNormalize: the env seed, `torch.initial_seed()` and numpy's state equal 6271's;
- `cell_problems`: a clean cell passes; a dirty tree, a changed or a missing file, and another head are each refused,
  and all four are listed together;
- `--json` overwrite: refused.

The short measurement on the smoke's pair now also asserts the record's `seeding`.

Measured [X/out/mutation3_run.log.txt, mutation3_run_rerun.log.txt; the method of review3/mutate.py, rewritten
where code moved]:

- V2, V3, V5, V6, V8, V9, V11, VC1 and VC2 are all KILLED.
- So is V11b, build() not seeding (`seeding = {}`), killed by the short measurement.
- V11c survives. It is a mutant that skips seeding but writes the right seeds into the record; only a comparison
  with a recorded run catches it.
- The FIX3 final run's 32-episode measurement of the smoke's 65,536 pair equals FIX2's in every value but
  `tool_sha256` (the new `seeding` aside) [measured: T/out/fix3_final/compare_with_fix2.txt]. So the refactor
  changed nothing measured, and the seeding is effective.

### Note 3: M4, the noise readout

- `pilot3_noise.py` (`e2a2fd58...` -> `95f05851...`) now wraps two reads: the `gate_progress.npz` load and reads, and
  the event-file reading.
- An input it cannot read prints that part "not run: ... could not be read (<error>)", and the other parts still
  print. The exit stays 0.
- The npz is now opened in a `with` block. The values are unchanged: the smoke's two runs give JSON equal to FIX2's
  but for `tool_sha256` [measured: compare_with_fix2.txt].
- Tests (7 -> 9, `3aebd816...` -> `6b25633d...`):
  - an npz cut in half, and an event record that passes its CRCs but cannot be decoded: each prints "not run" while
    the checkpoints still print;
  - an existing `--json`: refused.
- The review's own probe (review3/noise_cut_npz.py) now prints exit 0 in place of the crash [measured:
  T/out/fix3_final/noise_cut_npz.txt].

### Note 4: M5, the V - G text

- **(a)** The criteria's hidden-clock audit now states the n/a rule: "A comparison with fewer than 2 episodes reaching
  both buckets is n/a and leaves the bucket unflagged ... when fewer than 2 episodes reach step 800, [500,800) cannot
  be flagged".
- **(b)** By code reading [measured: T/tree environments/shared/curriculum/advancement.py, RewardRampCallback, and
  configs/tyrannosaurus_rex/anatomical/locomotion.toml lines 6 and 173-174]:
  - the forward weight ramps linearly from 0.2 to 2.0 over 2,000,000 steps, quantised to 0.001 and updated every
    10,000 steps; so 0.259 at 65,536;
  - `pilot3_vg.py` builds its envs with the stage's 2.0 and no ramp.

  So V and G share one reward at the four judged checkpoints (2,000,000-8,000,000), and the smoke's 65,536 check
  mixes 0.20-0.26 with 2.0. The text says so in the audit, and the smoke's V - G bullet says its [0,100) flag is read
  only roughly. That the smoke's override left the ramp alone is [inferred: the override changed the cadence keys
  only].
- **(c)** The audit and the tool's docstring now say that "mean V" and "mean G" pool the bucket's states, so their
  difference need not equal the episode-weighted mean V - G. The columns themselves are unchanged (report only;
  changing them would change a pinned output for no rule).

### Note 5: M6, the scorer

**(2) Fixed.** In `pilot3_score.py` (`b343c917...` -> `286ce04a...`), `habit_status` now reads a record with
`validation_only` true as NOT RUN, with the reason. `test_pilot3_score.py` (`1cfa8062...` -> `e7af7200...`) has 3
new checks:

- a validation-only NO HABIT record gives Q4 UNREAD, not YES;
- a validation-only HABIT record is not read either;
- `validation_only` false reads as before.

Result: 310/310 [measured: T/out/fix3_final/test_pilot3_score.log.txt]. Pilot 1's four outcome functions stay
byte-identical, and the scorer's function lists equal FIX2's [measured: T/out/fix3_final/fn_compare_score.txt].

**(1) Text only; code kept.** Why [measured by code reading: `seed_result`; the judge before FIX2
(SP/work/pilot3/fix2/before/pilot3_judge.py lines 341-343) and now both refuse "not an intact pair"]:

- The fallback (a binding without `intact_periodic_steps`) can only come from the judge before FIX2.
- No judged cell holds a broken pair, so the fallback never reads one. When the newest zip by file name is broken it
  has no judgement, and the seed reads missing ("has no valid judgement: judge it"). It is never read at another
  outcome.
- Removing the fallback would change pilot 1's adapted collapse case in `test_pilot3_score.py`, whose run_end has no
  intact list. The text promises that the adapted test changes only where it names checkpoints or the budget.

The criteria ("A seed's outcome", Collapse-stopped) and the scorer's docstring now say this.

### Note 6: M7

- All three report tools (`pilot3_vg.py`, `pilot3_noise.py`, `pilot3_habit.py` in every mode) refuse an existing
  `--json` target, exit 2, writing nothing. Each has a test, and the final run shows both refusals live
  [measured: T/out/fix3_final/habit_refuse_existing_json.txt, vg_refuse_existing_json.txt].
- `pilot3_habit.py` was added beyond the finding's two tools [inferred]: its record picks the Q4 branch, so a silent
  overwrite matters most there.
- The scorer's own `--json` (which writes score.json) is left as is: it was not named, and it does not take a
  per-cell path.
- **Not added: a branch-prefix check in `vg --cell`.** The cell's head must equal the tree's HEAD, and that pins the
  code that runs whatever the branch is called. The habit tool has no branch check either, and the judge's check
  already ran when the cell was made [inferred].

## Checks run in this stage

| Check | Result |
|---|---|
| Tool sha256s before any edit | 48/48 OK [measured: X/tools_sha256_check_before.txt]; criteria `2e1dc715...` as given |
| FIX3 final run, 21:59:08-22:02:44 UTC [measured: T/out/fix3_final/, its script T/out/fix3_final_run.sh] | every exit as expected (0, or 2 for the refusals); the tool files the same before and after; the tree clean after |
| pytest | 192/192: judge 85, habit 45 (34 + 11), runner 13, video 9, V - G 31 (25 + 6), noise 9 (7 + 2) |
| `test_pilot3_score.py` | 310/310 (307 + 3) |
| PC ledger validation | byte-identical to FIX2's |
| Function identity | `seed_outcome`, `pilot_outcome`, `reviewed_outcome` and `pilot_outcome_reviewed` byte-identical to pilot 1's; the scorer's lists equal FIX2's |
| Habit seed on FIX2's e2e cell | NO HABIT, equal to the e2e's record but for `window_check` and `tool_sha256`; the scorer on it reads PILOT OUTCOME STILL, Q4 NO by the pilot outcome |
| Habit refusals | existing `--json`, a stale judged copy, a same-named copy elsewhere: exit 2, nothing written |
| V - G on the smoke's 65,536 pair | every value equal to FIX2's run but `tool_sha256`; `seeding` 6271 for the envs, torch and numpy; an existing `--json` refused |
| Noise on the smoke's two runs; a cut npz | equal to FIX2's but `tool_sha256`; the cut npz prints the forward part "not run" (BadZipFile), exit 0 |
| Mutation check (X/bin/mutate3.py) | 4 baselines pass; 23 of 24 mutants killed; V11c (a faked seeding record) survives, as expected |
| ruff (F, E9, B) on the changed and judging files | only the 3 known B905 sites in pilot 1's scorer functions; E501 the same 24 lines as before FIX3 [measured: X/out/ruff_*.txt] |
| Criteria text format | ASCII only; no line over 120 columns but the two frozen Warm start lines |
| Re-pin | 73 entries (the 21 tool and test files and the evidence of the three final runs), 73 OK [measured: X/out/tools_sha256_check_after.txt]; the criteria cite all 21 current tool sha256s and no stale one [measured: X/out/cite_check3.txt] |

The e2e (`tests/e2e_parent_cell.py`) was **not** re-run in full, for three reasons:

- its video step starts Xvfb, which writes under the system temp folder (FIX2 recorded that as a slip);
- the judge and the video tool are unchanged;
- re-running it would replace FIX2's e2e evidence.

In its place, the parts it runs that FIX3 changed were run on FIX2's e2e cell: the habit seed mode and the scorer.
The e2e file itself is unchanged and pinned. Its FIX2 result (19/19) is still the evidence for the judge and video
steps.

## Tool sha256s, before and after

| File | Before (FIX2) | After (FIX3) |
|---|---|---|
| `pilot3_habit.py` | `eab5f398...` | `56f1c78deaea9ff64ef028eb10718304c944e90435a9b676b4c28f33855f1388` |
| `pilot3_score.py` | `b343c917...` | `286ce04a2c5f4e34b9f3e146431a14cbd40767050f83749f5968b9e270a97018` |
| `pilot3_vg.py` | `b633fbd1...` | `a58d5f4f24e0c3e70467c8aeea8df20188fc246d48b3127bef594061a6b44545` |
| `pilot3_noise.py` | `e2a2fd58...` | `95f058519c212236b95f6392a5ab24641eb80786932e77c1acaaaf9b6ae6bf33` |
| `test_pilot3_score.py` | `1cfa8062...` | `e7af72006bcf2ca99d7cfdbe7d328da3945c25b2ab95cfaa455fe9a6298ff597` |
| `tests/test_pilot3_habit.py` | `a9169dee...` | `a19fcd129d760177a0b2f39502618593a807f31a9ed6410508845b690e618945` |
| `tests/test_pilot3_vg.py` | `c13b3e12...` | `ead00c7b3a9cdb426cca297dc4b209e83dfbcc43d7f39ca55ec00cbba47b7913` |
| `tests/test_pilot3_noise.py` | `3aebd816...` | `6b25633d24719d7cea0a651abc43b7f34cf4480aecbfffb5b81ebba49e1143fe` |

Every other pinned tool file is unchanged. `tests/e2e_parent_cell.py` (`be256afc...`) is now cited in the criteria.

## Text files after this stage

| File | sha256 before | sha256 after |
|---|---|---|
| `criteria_pilot3.txt` | `2e1dc715...` (the input) | `6daef03df8094df222a9e692485f2d5e40c218270c9573e736ca31d24b3ecbc7` |
| `CHANGES.md` | `ce9c73f0...` | `ffcddfa019a542501cceac74bc4334d05c22781d4ca73b25f66f7d6caab6f702` |
| `NOTES.md` | `5967c5f0...` | `53de4c3a7020fbb801bf6de63acd6d06bc9593c000c119684bdbeaae7a2e23b1` |
| `BRANCH_PLAN.md` | `6b6ce6e4...` | `00906f09f54cb3b2f310926761bf78ac898727036e4c30bd496ac4531791c004` |

Diffs: X/out/diffs/. The untouched inputs are in X/before/.

## Still open (for the maintainer)

- **The FIX3 code needs a review**, as FIX2's did: the habit tool's path and window checks, the V - G refactor and the
  noise readout's guards.
- **The maintainer's choices, as before:**
  - approve the frozen text's sha256 (now the FIX3 one);
  - approve the three local commits made before that approval.
- **The pending list is unchanged:** the push with an explicit refspec, a durable copy of the tools (now the FIX3
  bytes), the Colab steps and their review, and the launch approval.
- **For the maintainer to note [inferred]:**
  - a run whose real ledger stops short of its deciding checkpoint can never be read by the habit tool; that seed
    reads NOT RUN, so Q4 reads UNREAD and never YES;
  - an incomplete last evaluation (fewer than 10 rows, for example a sync cut inside a row set) is not refused: the
    text reads it as not flagged, as before.

## Rule slips

- **Not every command went through the logger (X/r.sh).**
  - Logged by hand afterwards, with their times: the first listing call; the call that wrote r.sh and started
    CMDLOG.txt; one direct `sha256sum` of the changed files.
  - Shell heredocs wrote X/bin/stale_ledger_fixed.py, mutate3.py, compare_fix3.py, make_stale_cell.py, repin.sh,
    cite_check3.py and T/out/fix3_final_run.sh. One python heredoc edited mutate3.py. Each has a hand-written line
    with its time.
  - Not logged at all:
    - one call began with a stray fragment (the pilot3 folder's path run as a command), which failed with "Is a
      directory" and did nothing; the logged r.sh call after it ran as written;
    - four progress reads of T/out/fix3_final/rcs.txt and of the V - G output during the final run;
    - the watch (Monitor) on rcs.txt.
  - Edits and reads made with the file tools are not in CMDLOG.txt: every edit to the tool files, the tests, the
    criteria, CHANGES.md, NOTES.md and BRANCH_PLAN.md; the writing of this file; reads of the tools, the tests and
    parts of the criteria. The diffs of every edited file are in X/out/diffs/, and the untouched inputs in X/before/.
- **Simulations beyond the final run.** The mutation check ran the V - G short measurement (a 4-episode rollout on the
  smoke's pair) 3 times under mutants: V11b in the first full run, then V11b and V11c in the re-run. It ran one at a
  time, nice 10, OMP_NUM_THREADS=1, and nothing trained. These rollouts check the tests rather than re-run them, so
  they may go beyond "simulations only as the stage says".
- **One mutant was first mis-built.** In the first full mutation run, "V11b" faked the seeding record rather than
  removing the seeding, so it could not be killed. It was renamed V11c and kept, and V11b was redefined as `seeding = {}`
  and re-run (killed). The first run's results are kept as X/out/mutation3_results_first_full_run.json.
- **Other stages' files were used read-only.**
  - Scripts, run with `-I`, writing only into T/out/fix3_final and X/tmp: review2_code/bin/fn_compare.py and
    review3/noise_cut_npz.py.
  - The ruff binary of SP/work/optB/venv, with its cache in X/tmp.
  - Inputs: FIX2's e2e cell (T/e2e, read only; the stand-in stale cell was built in X/tmp) and FIX2's retired
    validation cell.
- **The system temp folder.** /tmp's own modification time changed during the stage. The only new entries there are
  the harness's own logs (claude-code*.log); none is this stage's [measured: find -newer X/r.sh]. No Xvfb ran.
  TMPDIR, pytest's `--basetemp`, MPLCONFIGDIR and XDG_CACHE_HOME pointed inside X; torch's inductor folder was
  made in X/tmp.
- **No pattern search for model names or device paths.** Writing either in a search pattern would itself break the
  rules, so, as at FIX2, every added line was read in the diffs instead.
- **What did not happen.** No training, no branch change, no commit, no push and no git config change. Nothing was
  written to /home/user/mesozoic-labs. The only writes outside X were the edits to T and FC and the new
  T/out/fix3_final/.
