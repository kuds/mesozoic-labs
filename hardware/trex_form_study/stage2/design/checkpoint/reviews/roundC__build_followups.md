<!-- Working paper, 16 Nov 2026: the builder's report on five small follow-ups to the bundle tool and the policy files. Nothing here is a rule. -->

FIVE FOLLOW-UPS ON THE BUNDLE TOOL AND THE POLICY FILES: all five done.

Simulation tooling only. No episode, fit or training ran. Worktree /home/user/mesozoic-wt/wpc-cb, branch wt/wpc-cb, tip f2e6aa7. Nothing pushed, no history rewritten. The main checkout was not touched (`git status --porcelain` there: 0 lines).

The working branch was merged twice. The first merge (a1c36b4) brought only new files. The branch then moved by nine commits while I worked, none touching the five files, so I merged again at the end (f2e6aa7). It is now 0 commits ahead of my tip.

1. PER ITEM

Item 1, the `checkpoints/` line (40d41f9).
- `stage2/tb/.gitignore` has 13 lines.
- `bundle.py` holds no list of the file's lines. The only list is `IGNORE_LINES` in the test, and it lost the line.
- In the test, `runs/r/jobs/j/checkpoints/checkpoint_1.json` moved to the paths git must not ignore. The `.zip`, `.pkl` and `.npz` under the same directory stay ignored, and I added `checkpoint_1.json.tmp` and `model_1.zip.part` there.
- Without the change, `test_the_ignore_file_holds_exactly_its_lines` and `test_git_check_ignore_on_every_output_path` failed (2 failed).

Item 2, `build` refuses on the study itself where git cannot tell (907975b).
- "The study itself" means the root resolves to `s2io.STUDY`, so a link or another spelling counts too.
- The refusal comes before anything is written and carries the git command and git's first line. Seen on the command line: `S2 BUNDLE probe: NOT BUILT: bundle probe is not built: the root ... is the study itself and git cannot tell whether the records read are committed there (git rev-parse --is-inside-work-tree: fatal: not a git repository: ...)`, exit 1.
- Another root keeps the note and `records_committed: null`; the note now carries git's reason.
- `provenance.root` reads "." for the study under any spelling.
- Test: `test_a_build_of_the_study_itself_is_refused_where_git_cannot_tell` covers no work tree, no git, a refused checkout and a failing listing, the same four as a note on another root, and a successful build once the records are committed. The stand-in test also runs the command line on this checkout with `GIT_DIR` pointing nowhere.
- Without the change: "DID NOT RAISE BundleError", and the stand-in build exited 0.

Item 3, `manifest_problems` cross-checks the tables (684ce25).
- New `_table_problems` and `CONTENT_KEYS`. These are now problems:
  - `bodies` is not a sorted, non-empty list of distinct names.
  - `models` does not hold one entry per body with exactly `case`, `xml`, `record`.
  - A model path is not `<case>/models/<id>.xml` with `<id>.json` beside it.
  - A model file is missing from the file list, has another sha256 there, sits in another part than `models-<body>.tar`, or belongs to two bodies.
  - A set does not hold one entry per body with exactly `freeze_index`, `manifest_sha256`, `files`, or its count is not the number of listed files under `stage2/sets/<set>/<body>/`.
  - A file is neither a model file nor a set's `refs/` or `transitions/` `.npz`.
- `verify` reports these and `unpack` refuses before it writes.
- Tests:
  - `test_junk_in_the_manifests_own_tables_does_not_verify`: 38 kinds of junk with the hash refitted, each with its sentence.
  - `test_any_junk_anywhere_in_the_tables_is_a_problem_and_never_another_exception`: every place in the three tables given each JSON kind, removed, or given one more key.
  - `test_a_part_holds_the_files_of_one_owner`.
  - `test_what_the_tables_cross_check_cannot_tell`.
  - Every manifest the other tests build has no problem.
- Without the change: 37 failed (35 junk cases at that point, the sweep, and the positive check).

Item 4, `actions_at_bound` (6490a0b).
- `s2load.py`: the record's fingerprint block gains `"actions_at_bound": int(np.count_nonzero(np.abs(actions) >= 1.0))`, the review's formula.
- The schema name stays `s2-ckpt-2`. Nothing is refused on the count, and a record without the key still loads (tested).
- Tests: the assertion in `test_the_record_and_the_four_files` (0 for the stub policy, plus the block's exact keys), and a new `test_the_record_counts_the_fingerprint_actions_on_a_bound`. The new test scales the output layer by 300 and compares the count with the policy's own output before the clip; more than a quarter of the entries are on a bound, and the checkpoint loads with difference 0.0.
- Without the change: both failed with `KeyError: 'actions_at_bound'`.

Item 5, single faults. I put 37 single faults for items 1 to 3 into a scratch copy. The first pass missed 3; I closed those gaps in the tests and re-ran the three, which were then caught. Three wrong ways to count for item 4 each make the new test fail.

2. TEST SUMMARY LINES (tip f2e6aa7, the given environment, through `bash tools/lowprio.sh`)
- `tests/test_stage2_bundle.py`: `118 passed in 11.01s`, exit 0 (76 before).
- `tests/test_stage2_s2load.py tests/test_stage2_schedules.py tests/test_stage2_tbenv.py`: `63 passed in 24.72s`, exit 0. No skip: the test that waited for the ignore file now asserts, and passes with 13 lines because the live record under `work/` is still ignored.
- Hash command: `45719d030359ec7d5d3914af3814d837fdc05d77652ba45c64792b511027528d c0027992de21067525cc4489b74acb157feb89f3a4c7618580addc2b341be1c1`, the registered pair.
- `ruff check` on the four Python files: `All checks passed!`

3. `git diff claude/mesozoic-trex-stage2-refine-4jffre --stat`
```
 .../trex_form_study/designlab/stage2/bundle.py     | 227 ++++++++++++--
 .../trex_form_study/designlab/stage2/s2load.py     |  16 +-
 hardware/trex_form_study/stage2/tb/.gitignore      |   1 -
 .../trex_form_study/tests/test_stage2_bundle.py    | 337 ++++++++++++++++++++-
 .../trex_form_study/tests/test_stage2_s2load.py    |  33 ++
 5 files changed, 568 insertions(+), 46 deletions(-)
```
`git status --porcelain`: empty.

4. FOR THE LEAD

- Item 3 goes further than the three tables in three places. Each is a line or two to drop; I recommend keeping all three, since `build` never writes otherwise.
  - A key in `content` other than the six is a problem.
  - A part that holds no file, or files of two owners, is a problem.
  - A set's files, in sorted order, must fill `<set>-<body>-<i>.tar` from 1 on without a gap.
- What the cross-check cannot tell, held by a test: another `freeze_index`, another 64-hex `manifest_sha256`, a set entry that lists no file, another `bundle_id`. The notebook's comparison of the quoted id and content hash stays the defence; it could also compare `content.sets[set][body].manifest_sha256` with the clone's latest freeze.
- Not added: a check of model paths against the registry (`bundle.model_paths(body)`). It would tie `verify` to the verifying checkout's registry. Recommendation: leave it out.
- Item 2: a root passed as `root=` that is another tree than the study and no git checkout still builds with a note and null, as you ruled. The command line has no `--root`, so it always refuses there.
- Item 4:
  - `ROUND_C_DESIGN.md` section 3.6 (the `checkpoint_<n>.json` line) does not list `actions_at_bound`. That is your file.
  - No threshold is set for "mostly saturated"; whether the monitor or the report flags one is your choice.
  - A later reader of the fingerprint block (trainer, monitor) should use `.get`, since older records lack the key.
- Decision D22's "14 lines" and the bundle brief still need their text changed to 13.
- All six commits end with the two attribution lines, the two merges included, so a plain merge of the tip is possible. Otherwise cherry-pick 40d41f9, 907975b, 684ce25, 6490a0b.

5. NOT DONE / LIMITS
- No full test suite; only the four test files named.
- The refused-checkout and failed-listing cases are simulated by replacing `subprocess.run`; no real "dubious ownership" checkout was made.
- I did not run the three policy test files before my change, so "63" has no before-count of mine.
- Scratch leftovers: /tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/cb_followup/ (`copy/`, `mutate.py`, `mutate.out`, `mutate2.out`, `mut/test_mut_s2load.py`).

6. COMMITS, oldest first
a1c36b4ea905a62b871fe08fdee7513ccf25dabd  Merge the working branch into wt/wpc-cb (the worker pool, the rulings of 16 Nov, the round C reports)
40d41f938e0dfddcde802d05699e907b83753bdc  Track B's ignore file: the line checkpoints/ goes, a checkpoint's record is committed
907975bceb6d95f4d451d03bb921895d64eeb39e  Bundle: a build of the study itself is refused where git cannot tell
684ce2502ba7da1e53ba132449180b4aed159af6  Bundle: manifest_problems checks the tables of bodies, models and sets against the file list
6490a0bc2bf6f35f5c162cd47b3b6897bb0676f1  Checkpoint record: the fingerprint block counts the actions on a bound
f2e6aa7e1c6e11158c8d9459381ff16d2b5715f9  Merge the working branch into wt/wpc-cb (the env's small fix round, the added rulings, the servo model note)
