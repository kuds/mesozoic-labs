# fix:CB (roundC, wave 2a): the agent's final report, 16 Nov 2026

Working paper. Nothing here is a rule.

WORK PACKAGE CB (Drive bundle tool) - fixer's report

Simulation tooling only: no episode ran, nothing was uploaded, no network call. Worktree /home/user/mesozoic-wt/wpc-cb, branch wt/wpc-cb. The main checkout was not touched (`git -C /home/user/mesozoic-labs status --porcelain`: 0 lines before and after). Nothing pushed, no history rewritten.

Merge first: the working branch had moved by three commits (b134773, b57fdd0, 82ba1ad: servo-model notes, checkpoint status). Merged without conflict as 888ab67, this time with the two attribution lines. It has not moved since (`git log HEAD..claude/mesozoic-trex-stage2-refine-4jffre`: 0 lines). Record-key hashes at the tip: `45719d030359ec7d5d3914af3814d837fdc05d77652ba45c64792b511027528d c0027992de21067525cc4489b74acb157feb89f3a4c7618580addc2b341be1c1` (the registered pair).

1. PER FINDING

Finding 1 (`checkpoints/` hides `checkpoint_<n>.json`): FOR THE LEAD, code unchanged. Confirmed: `git check-ignore -v stage2/tb/runs/r/jobs/j/checkpoints/checkpoint_1.json` gives `.gitignore:13:checkpoints/`. The brief and D22 give the 14 lines "exactly"; the design's layout and CN's item 8 want that record committed. See "for the lead" 1.

Finding 2 (merge commit 5aaf021 without the attribution lines): FOR THE LEAD. I cannot change it without rewriting history. See "for the lead" 2 for ways to keep it out of the working branch.

Finding 3 (CN sees only the printed line; 12 hex of the hash): FIXED. `bundle.py` `main`: `verify --json` and `unpack --json` print the returned result as one line of JSON and nothing else; a refused unpack prints `{"first_problem": ..., "ok": false}`. Exit codes and the default lines are unchanged. Test: `test_cli_json`. It is an addition to the frozen CLI: "for the lead" 3.

Finding 4 (test gaps): FIXED, tests only (commit d8e23b4; the module passes them unchanged).
- M16: a same-size file with one bit wrong, in `test_unpack_gives_the_files_and_twice_is_a_no_op`.
- M34: `test_the_order_of_the_names_given_does_not_matter` (bodies and sets reversed: same content, same part bytes).
- M27: `test_a_file_that_changes_while_it_is_packed_is_refused` (monkeypatched `_write_part`).
- M24: `test_a_part_that_changes_while_it_is_unpacked_is_refused` (monkeypatched `scan_part`).
- M25: not an equivalent mutant for every name. `test_a_name_ustar_cannot_hold_is_refused` (a 120-character file name) fails under PAX; GNU already failed `test_tar_rules`.

Finding 5 (tracebacks instead of refusals): FIXED, all four bullets confirmed first (JSONDecodeError, KeyError, AttributeError, OSError 107, RecursionError), all now a `BundleError` with the path, or a reported problem in `verify`.
- `_set`, `_read_record`: a set record that is not JSON, no object or nested too deep; a latest freeze of another shape; a manifest or record table that is a list. Tests: 11 new cases in `test_build_refuses_and_writes_nothing`.
- `unpack` and `_dest`: a part or a place that cannot be looked at or read. Tests: `test_a_part_that_cannot_be_read_is_reported_and_refused`, `test_unpack_refuses_a_place_it_cannot_look_at`.
- `verify` (`check`): `is_file` is now inside the try. Same first test.
- `load`, `manifest_problems`, `verify`: a manifest nested too deep to parse (100,000) or to hash (3,000). Test: two new cases in `test_verify_never_raises_on_a_bad_manifest`, which now also runs `unpack` on every case.
- Same class, found while checking, also fixed: an inputs-list entry that is no sha256 (`_model`, was TypeError); a file at MANIFEST.json whose content block is no table (`_build`, was AttributeError; test `test_build_refuses_a_file_that_is_no_manifest_at_its_place`).

Finding 6 (`--only ""` says OK): FIXED. `unpack`: an empty selection is refused, and so is a selection that is no list of names (a bare string was read letter by letter; `[3, ...]` raised TypeError). Tests: `test_unpack_only_the_named_parts`, `test_cli_verify_and_unpack` (exit 1, `S2 BUNDLE: NOT UNPACKED: no part is named`).

Finding 7 (stale temporary parts): FIXED. `_build` removes every `*.tar.tmp` directly in the parts directory under the lock; `_write_part` creates its file with `O_EXCL`. I did not use `s2io.sweep_tmp`: it is recursive and leaves links. Found on the way and confirmed on the builder's module in a scratch copy: a link standing at a temporary part name was written through (the target became 51,200 bytes of tar) and moved into place as a link. Test: `test_a_build_removes_the_temporary_parts_a_killed_build_left`.

Finding 8 (`records_committed: null` where git cannot tell): FOR THE LEAD, behaviour unchanged. `build`'s docstring now says what null means. See "for the lead" 4.

Finding 9 (optional):
- a) `manifest_problems` checks files and parts only: NOT CHANGED. The function does what its contract says and the reviewer calls it optional; the defence is CN's comparison ("for the lead" 5).
- b) `provenance.commit` is the tool's checkout: not a defect (the CLI has no `--root`); now stated in `build`'s docstring.
- c) `unpack` takes no lock: not changed, stated in its docstring. A lock file under the root would be an untracked file of a clone (`.lock` is ignored only under `stage2/tb/`).

Evidence that each test fails without its fix:
- Before any change to the module, the new tests gave 18 FAILED lines (I did not capture that run's summary line).
- 28 single faults, each one fix or re-check removed, put into a scratch copy of the committed files: `28 faults, 0 not caught`. The unchanged scratch copy gave `73 passed, 3 skipped in 7.12s`. The worktree was never mutated.

2. TEST SUMMARY LINES (worktree's study directory, the given environment, through `bash tools/lowprio.sh`, tip 716580e)
- `pytest -p no:cacheprovider tests/test_stage2_bundle.py -rs`: `76 passed in 9.71s`, exit 0 (37 test functions; before the fix `56 passed in 7.30s`).
- Each commit's state on its own: 60, 74, 74, 75, 76, 76 passed.
- `ruff check` on both files: `All checks passed!`
- Acceptance, outputs in the scratch directory:
  - build: exit 0, `S2 BUNDLE standin: 2 bodies (60-N-T-10, 75-M-H-10), no set: 4 files in 2 parts, 1986560 bytes`; part hashes 65ac70ea...1cae and 2e9a55fc...6917, content sha256 eaeb1bcc4b76d246de44fa791ebb9d185391703065650504b73e85aa9da408a2: the builder's and reviewer's values, so the part bytes did not change. The manifest read `dirty False`, `records_committed True`.
  - verify: `S2 BUNDLE standin: manifest eaeb1bcc4b76, 4 of 4 files match their sha256, 2 of 2 parts: OK`, exit 0.
  - unpack: `4 written, 0 replaced, 0 already there: OK`, exit 0; again `0 written, 0 replaced, 4 already there: OK`; verify of the unpacked root `4 of 4 files match their sha256: OK`.
  - `verify --json`: `{"bundle_id": "standin", "content_sha256": "eaeb1bcc...08a2", "files": 4, "files_ok": 4, "first_problem": null, "ok": true, "parts": 2, "parts_ok": 2, "problems": []}`, exit 0.
  - `unpack --only ""`: exit 1.
- `git check-ignore -v`: `work/x`, `bundle/x/parts/a.tar`, `a.tar.tmp`, `bundle/x/.lock`, `MANIFEST.json.tmp` ignored; `bundle/x/MANIFEST.json`, `runs/r/jobs/j/run.json`, `gates/V6/b/g8a.json` not (exit 1).

3. DIFF AND STATUS
`git diff claude/mesozoic-trex-stage2-refine-4jffre --stat`:
 .../trex_form_study/designlab/stage2/bundle.py     |  980 +++++++++++++++
 hardware/trex_form_study/stage2/tb/.gitignore      |   14 +
 .../trex_form_study/tests/test_stage2_bundle.py    | 1271 ++++++++++++++++++++
 3 files changed, 2265 insertions(+)
`git status --porcelain`: empty.

4. COMMITS OF THIS ROUND (oldest first; each ends with the two attribution lines, checked by script; no other mention of a model)
888ab672a8861452608ae5b6b71ca0eacbe70c8a  Merge the working branch into wt/wpc-cb (servo model notes, checkpoint status)
d8e23b4a85b90c6d16896934ec5de98193fd7266  Bundle tests: the gaps the review's single faults showed
ddba0d607f37e8f1e1c8affb6e7e765d605077eb  Bundle: a record, manifest, part or place that cannot be read is a refusal
eb2c5c1570f64dc1ad4e2303c8d43ebf6c095c14  Bundle: unpack refuses an empty selection of parts
ccaacfb54badc7dc580539b66efff72e074e6638  Bundle: a build removes the temporary parts a killed build left
ce8f25d9a7c1273fbe578cb5d11bd9cbbd7fc544  Bundle: verify and unpack print their result as one line of JSON with --json
716580ef5e6496c1f2d3c1163fece5f1d14a5129  Bundle: what provenance.commit and records_committed say; unpack takes no lock
Head of wt/wpc-cb: 716580ef5e6496c1f2d3c1163fece5f1d14a5129. The six package commits touch only `bundle.py` and the test file.

5. FOR THE LEAD
1. The `checkpoints/` line (finding 1), needed before CN.
   - (a) Drop the line. The binaries stay ignored by `*.zip`, `*.pkl`, `*.npz`, `*.tar`, `*.part`, and live jobs by `work/`.
   - (b) Keep it; `nb collect` puts the records under another directory name.
   - (c) Keep it; commit with `git add -f`.
   - I recommend (a). It needs D22 and the brief amended to 13 lines. In CB it is three edits: line 13 of `stage2/tb/.gitignore`; `IGNORE_LINES`; `runs/r/jobs/j/checkpoints/checkpoint_1.json` moved from IGNORED to NOT_IGNORED in the test.
2. The unattributed merge commit 5aaf021 (finding 2). The reviewer's "merge 7f01dda" no longer brings the fixes, which sit above it.
   - (a) Merge the tip: 5aaf021 enters the working branch's history.
   - (b) `git merge 7f01dda`, then `git cherry-pick d8e23b4 ddba0d6 eb2c5c1 ccaacfb ce8f25d 716580e`.
   - (c) Cherry-pick all eleven package commits: caaf14c 085b69f 7999abf 55564c7 7f01dda, then the six above.
   - (d) `git merge --squash`.
   - I recommend (b). Seen: both merge commits change none of the three files (`git diff --stat` empty), each of the eleven commits touches only those files, and `git merge-tree --write-tree <working branch> HEAD` gives the tip's own tree 724f7d8b07ce5447a9230c2e7fa6a451bf79bf86. I did not run the cherry-picks.
3. `--json` is outside the frozen CLI, as `--only` was. Alternatives: CN parses the printed line, or reads the 64 hex from the committed manifest after a verify that says OK. CN's brief does not know the flag yet.
4. `records_committed: null` (finding 8).
   - (a) As now: a printed note and null; you check that the real manifest reads `records_committed: true` with a non-null `commit` before committing it.
   - (b) `build` refuses when the root is the study itself and git cannot tell.
   - (c) Refuse everywhere; that needs an opt-out for the synthetic test trees, so an API change.
   - I recommend (b): the CLI always builds from the study, and the brief says "from committed records only". Not done, because it changes when a build is refused.
5. Junk in `content.bodies`, `models` or `sets` with a refitted hash still verifies (finding 9a). CN should compare `bundle_id` and the 64-hex `content_sha256` from `verify --json` with the quoted ones, and could compare `content.sets[set][body].manifest_sha256` with the clone's latest freeze. A cross-check of the manifest's own tables in `manifest_problems` is possible in CB if you want it.
6. Readings of mine:
   - An empty `parts` selection is refused in the API too, not only on the command line.
   - The sweep removes links named `*.tar.tmp` as well (the link, never its target).
   - `verify`'s wording for a part or file that cannot be looked at is `<name>: cannot be read (<os error>)`.
   - A refused `unpack --json` has the keys `ok` and `first_problem` only.

6. NOT DONE / LIMITS
- No change to `stage2/tb/.gitignore`, to any file outside the package, or to the behaviour behind findings 1, 2, 8 and 9a.
- The full test suite was not run (forbidden); only `tests/test_stage2_bundle.py`.
- Still Python 3.13.18 only; no Colab, no mounted Drive, no V6 bundle, no upload.
- The "mount that went away" and "place that cannot be searched" cases are simulated by monkeypatching `Path.is_file`, `bundle._hash` and `Path.is_dir` (ENOTCONN, EACCES), not with a real mount.
- The 3,000-deep manifest case depends on the Python version for where it fails (parser or hash); both are caught.
- `.ruff_cache/` at the worktree root (git-ignored, dated 14:21, before my first command) was there already; my lint runs updated it.
- Left in the scratch directory /tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/fix_cb/: `probe1.py`, `probe_link.py`, `mutate.py`, `split.py`, `red.txt`, and the per-commit file states under `stages/` and `final/`. The scratch study copy `mutate.py` ran on was deleted; it needs a copy of `designlab`, `tests`, `tools`, `stage2/tb/.gitignore` and the four inputs lists.
