<!-- Working paper, 16 Nov 2026: the independent review of those follow-ups. Nothing here is a rule. -->

REVIEW OF THE FIVE FOLLOW-UPS (wt/wpc-cb, tip f2e6aa7)

Simulation tooling only. I ran no episode, fit or training. Nothing here is a result on the robot. Every number below is from my own tool output or my own count.

VERDICT: merge after one fix, and the fix is on the merge itself. The code can go in as it stands, but only by cherry-pick of 40d41f9, 907975b, 684ce25, 6490a0b in that order. Do not merge the tip.

FINDINGS, most serious first

1. A plain merge of the tip would bring in the commit the lead ruled out. (Process; blocks a plain merge.)
- Evidence: `git log claude/mesozoic-trex-stage2-refine-4jffre..wt/wpc-cb` lists 14 commits, not six. One of them, `5aaf021` ("Merge branch 'claude/mesozoic-trex-stage2-refine-4jffre' into wt/wpc-cb"), ends with its subject line only. Ruling CB-2 (`ROUND_C_RULINGS_16NOV.md` lines 15-16) keeps it out of the working branch.
- The other 13 end with the two lines, the merges 888ab67, a1c36b4 and f2e6aa7 included.
- The builder's sentence "all six commits end with the two attribution lines, so a plain merge of the tip is possible" is wrong.
- Fix: cherry-pick the four commits. Checked in a scratch clone on 355e91a: each applies without conflict, and the final tree is 81d2bc42..., the tip's own tree. Each touches only its two files, and each ends with the two lines.

2. No test holds the `.npz` condition on a set's file name. (Minor, test gap.)
- Where: `designlab/stage2/bundle.py` lines 749-750.
- Evidence: with `names[-1].endswith(".npz") and names[-1] != ".npz"` removed, the whole of `tests/test_stage2_bundle.py` still passes.
- The code itself is right: my probe with `refs/notes.txt` and with `refs/.npz` (counts refitted) gives "neither a model file ... nor a set's file".
- Fix: one more JUNK case, a set's file that is no `.npz`.

3. No test holds that only sound file entries reach the table check. (Minor, test gap.)
- Where: `bundle.py` line 845, `_table_problems(content, good, sound)`.
- Evidence: with `good` replaced by `files`, the whole test file still passes. With that fault a file entry that is no table would raise inside `verify`.
- The code itself is right: my sweep over every place of the whole content block (1,626 single changes, hash refitted) raised no other exception, and `unpack` refused with `BundleError` only.
- Fix: extend the sweep in `test_any_junk_anywhere_in_the_tables_is_a_problem_and_never_another_exception` to `files` and `parts`, asserting "never another exception".

4. One branch cannot be reached. (Minor.)
- Where: `bundle.py` lines 738-740, the models-side "its part ... also holds files of".
- Evidence: made silent, the whole test file still passes. At that point `owner` holds only `models-<body>.tar` keys of other bodies, so the comparison is never unequal for a JSON manifest.
- Fix: replace by a plain `owner[e["part"]] = ("models", body)`, or leave it.

5. For the lead: where the changes go further than the rulings. None blocks the merge.
- The builder's three extras in item 3: a content key outside the six is a problem; a part with no file or with two owners is a problem; a set's files must fill parts 1, 2, ... without a gap. Options: keep or drop each. Recommendation: keep all three.
  - `build` never writes otherwise.
  - All 8 registered bodies and the 3 frozen sets on disk (F25F9, F25F9cl, V0 of 75-M-H-10) have the forms the check asks (my probe).
  - `test_a_real_frozen_set` ran and passed in the builder's worktree.
- Not named by the builder: an entry of `content.files` or `content.parts` may still hold another key with no problem (30 of 30 such changes in my sweep). The docstring's "nothing is added to the content after the build" is therefore not enforced there. Options: leave it, since ruling 5 names bodies, models and sets only; or make the two entry shapes exact (`bundle.py` lines 826-827 and 838-839, plus two JUNK cases). Recommendation: make them exact at the next change to the file if the content-keys extra stays.
- Item 2 extras: `provenance.root` reads "." for the study under any spelling, and the note for another root carries git's reason. Recommendation: keep.
- Other things `manifest_problems` alone still accepts in my sweep: another `bundle_id`, `freeze_index` 0, and `bytes` 0 in a file or part entry. `verify` against the files catches the sizes.

6. The lead's papers still to change: `ROUND_C_DESIGN.md` line 422 lists `checkpoints/` among the ignore lines; lines 1108-1110 do not list `actions_at_bound`; D22's "14 lines"; the bundle brief.

WHAT I CHECKED AND FOUND RIGHT

- Only the five files changed: `git diff ... --stat` at 355e91a gave the builder's table (568 insertions, 46 deletions). `git status --porcelain` is empty in the worktree and in the main checkout.
- The working branch moved during my review to 09f20a0, one commit on three papers, none of the five files. The two-dot diff now also shows those three; the three-dot diff shows the five. `git merge-tree --write-tree` gave no conflict before (tree 81d2bc42...) and after (tree aa36c95b...).
- Record-key hashes, printed in the worktree: bar 45719d030359ec7d5d3914af3814d837fdc05d77652ba45c64792b511027528d, panels c0027992de21067525cc4489b74acb157feb89f3a4c7618580addc2b341be1c1. These are the registered pair.
- `ruff check` on the four Python files: `All checks passed!`
- Item 1 is the ruling and no more. The file has 13 lines. 4 of 4 single faults were caught; with the line put back, `test_the_ignore_file_holds_exactly_its_lines` and `test_git_check_ignore_on_every_output_path` fail. No other test on any branch reads this file's lines.
- Item 2 is the ruling, with the two small extras above. 12 of 12 single faults were caught. I also ran the real command line in a scratch clone under three real conditions. Each gave exit 1 and no file written:
  - no git on the PATH: "git could not be run (No such file or directory)";
  - a checkout owned by another user: "git rev-parse --is-inside-work-tree: fatal: detected dubious ownership in repository at ...";
  - a spoiled index: "git ls-files: fatal: .git/index: index file smaller than expected".
  The plain build there gave exit 0.
- Item 3: 44 of 47 single faults were caught; the three missed are findings 2 to 4. With the whole check off, the junk cases, the sweep and the one-owner test fail. `unpack` refuses before it writes.
- Item 4 is the ruling: the review's formula, schema name unchanged, nothing refused on the count. 9 of 9 single faults were caught, among them `>` for `>=`, upper bound only, rows for entries, a load that reads the key, and a load that refuses on the count. The action bounds are +-1 in `r5env.py` line 280 and `s2load.py` line 632.

`actions_at_bound` AND OLDER FILES

- No Track B checkpoint exists on disk in the main checkout or any worktree. `stage2/tb/` holds only `.gitignore`.
- A load reads only `obs_sha256` from the fingerprint block (`s2load.py` line 665). No other module reads the block, on this branch or on wt/wpc-ct, -ck, -ce, wp-l, prereg-b.
- Tested: yes, for the inference load. `tests/test_stage2_s2load.py` lines 385-389 delete the key, reseal the record and load it.
- Not tested by the suite, checked by my probe: a checkpoint written by the loader as it is on the working branch has no such key, and the new loader took it through `pair_problem` (None), `newest_intact`, the inference load (difference 0.0), the training load (0.0), `pack` and `import_tar` (ok). The other way round, the old loader loads a new record (0.0).

EXACT TEST SUMMARY LINES (worktree, tip f2e6aa7, the given environment, one process at a time)
- `tests/test_stage2_bundle.py`: `118 passed in 11.45s`
- `tests/test_stage2_s2load.py`: `44 passed in 22.22s`
- `tests/test_stage2_schedules.py`: `8 passed in 1.28s`
- `tests/test_stage2_tbenv.py`: `11 passed in 0.71s`
No skip. On the working branch the four files collect 76, 43, 8 and 11 tests.

NOT CHECKED
- The full test suite; only the four files.
- My single faults ran in a scratch clone where `test_a_real_frozen_set` skips (no set arrays there), so they were not run against it. Baseline there: `117 passed, 1 skipped`.
- The builder's own 37 faults and scratch leftovers; I ran my own 72.
- Any Colab or Drive step, and any real bundle manifest: none exists on disk.
- The count on a trained policy: stub policies only.
- The other worktrees' tests with these changes merged.

Files are in /tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/review_followups:
- mutate.py
- item3.out
- item3b.out
- item4.out
- probe_tables.py
- probe_ckpt.py
- bundle.diff
- bundle_test.diff
- s2load.diff

The scratch clone is removed.
