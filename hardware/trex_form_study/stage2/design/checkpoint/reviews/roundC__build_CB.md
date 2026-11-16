# build:CB (roundC, wave 2a): the agent's final report, 16 Nov 2026

Working paper. Nothing here is a rule.

WORK PACKAGE CB (Drive bundle tool) - builder's report

Worktree /home/user/mesozoic-wt/wpc-cb, branch wt/wpc-cb. Everything is simulation tooling: no episode ran, nothing was uploaded, no network call was made (the tool only starts local `git` subprocesses). The main checkout was not touched (`git -C /home/user/mesozoic-labs status --porcelain`: 0 lines before and after).

1. MERGE OF THE WORKING BRANCH
- At the start: `git merge --no-edit claude/mesozoic-trex-stage2-refine-4jffre` was a fast-forward, 1aefbd0 -> 26b057e (233 commits, no merge commit, no conflict).
- At the end the working branch had moved by two commits (4f2b33d, bb5b63d: `stage2/design/TRACKB_RULINGS3_16NOV.md`, `STAGE2_75_PLAN.md`). Merged again, no conflict, merge commit 5aaf021. I read the new rulings: none concerns the bundle (N3, N4) or the ignore file.
- MY ERROR: 5aaf021 carries git's default message ("Merge branch '...' into wt/wpc-cb") and lacks the two attribution lines (I used `--no-edit`). I did not amend it, because the rules forbid rewriting history. It is the branch tip, so one `git commit --amend` by the lead fixes it if wanted.
- Record-key hashes, checked after the first merge and again at the final tip: `45719d030359ec7d5d3914af3814d837fdc05d77652ba45c64792b511027528d c0027992de21067525cc4489b74acb157feb89f3a4c7618580addc2b341be1c1` (the registered pair).

2. WHAT I BUILT
a) `stage2/tb/.gitignore`: exactly the 14 lines of the brief, no comment line.
   Tests: `test_the_ignore_file_holds_exactly_its_lines`; `test_git_check_ignore_on_every_output_path` (scratch repo holding only this file at stage2/tb/: the brief's 10 ignored and 3 not-ignored paths plus live files, checkpoint files, lock and temporary names; also with the files really there, `git status` lists the committed kind only); `test_a_build_under_stage2_tb_leaves_only_the_manifest_for_git`.
b) `designlab/stage2/bundle.py` (904 lines): `build`, `content_sha256`, `verify`, `unpack`, CLI `build|verify|unpack`; helpers `load`, `manifest_problems`, `scan_part`, `model_paths`, `path_problem`, `tar_bytes`; `BundleError(s2io.S2Error)`.
   - build: per body the two files plant.model_files names (paths from the Stage 1 registry; the plant is never imported, a missing file is refused, nothing is rebuilt); per set and body the files of `setio.frozen(set.json)` (latest freeze), never look twins or unlisted files. Collects every difference, then refuses with all of them and writes nothing. Parts are written to temporary names, read back member by member, then moved; MANIFEST.json (schema s2-bundle-2, `s2io.write_json`) last. Tar rules as the brief gives them.
   - verify: never raises (an unreadable manifest is a reported problem). unpack: checks every part and every member before the first write; idempotent.
   - Imports: standard library, numpy, s2io, the `designlab.stage1` registry; setio only inside build when a set is packed.
   Tests (`tests/test_stage2_bundle.py`, 28 functions, 56 cases, all under tmp_path, mostly on a synthetic tree with two synthetic bodies and a twice-frozen set):
   - two processes, second under umask 077, other hash seed, time zone and cwd: byte-identical parts, equal content hash (`test_two_processes_and_another_umask_give_the_same_bytes`); `test_tar_rules`, `test_tar_bytes`.
   - content hash unchanged under CRLF, other indentation, compact form, reversed key order, changed/added/removed provenance; changed by one content character (`test_the_content_hash_ignores_layout_and_provenance`).
   - what is packed: models + latest freeze only (`test_a_bundle_holds_the_models_and_the_latest_freeze_and_nothing_else`, `test_sets_empty_packs_the_models_only`, `test_the_ledger_hash_where_no_record_holds_it`).
   - split: `test_a_set_over_the_part_size_splits_and_every_file_is_in_exactly_one_part`.
   - build refusals, each with its path and nothing written: 14 parametrized cases (`test_build_refuses_and_writes_nothing`), `test_build_refuses_bad_arguments`, `test_a_second_build_leaves_the_manifest_and_another_content_is_refused`, `test_records_must_be_committed`.
   - verify: `test_verify_counts_and_reports_each_difference_with_its_path` (changed file, missing file, truncated/missing/bit-flipped part), `test_verify_never_raises_on_a_bad_manifest` (15 bad manifests).
   - unpack: `test_unpack_gives_the_files_and_twice_is_a_no_op` (mtime and inode unchanged on the second call); 16 parametrized refusals in `test_unpack_refuses_before_it_writes_anything` (truncated part, one changed bit, missing part, extra member, `..`, `..` inside a path, absolute member, symbolic link, hard link, directory, other bytes, member of another part, member twice, member short, no tar, gzip), each asserting no file was written; `test_unpack_refuses_a_place_that_resolves_outside_the_root`; `test_unpack_only_the_named_parts`.
   - CLI and imports: `test_cli_verify_and_unpack`, `test_verify_and_unpack_import_neither_mujoco_nor_setio`.
   - the study's own files: `test_the_lists_and_the_sets_directory_are_the_studys`, `test_model_paths_are_plant_model_files`, `test_a_stand_in_bundle_of_the_picks_models` (CLI on the real picks), `test_a_real_frozen_set` (skips without the sets' arrays).
   - Nine deliberate faults put into bundle.py one at a time (mtime 5; no `..` check; no scan before writing; no realpath check; provenance hashed; look twins packed; members unsorted; right files rewritten; part hash unchecked): each made the test file fail; the module was restored byte for byte (`cmp`).

3. ACCEPTANCE (from the worktree's study directory, the given environment; every command through `bash tools/lowprio.sh`, nice 19, in place of the brief's `nice -n 15`; final tip 5aaf021; load average 0.2-0.5)
Setup: `$PY tools/restore_inputs.py copy-from /home/user/mesozoic-labs/hardware/trex_form_study --group picks-models picks-arrays` -> "picks-models 6 6 0 0 agree", "picks-arrays 84 84 0 0 agree", "all here with their listed sha256". I also copied `--group sets` (67 files, "agree") so that the real-set test runs; the brief's setup does not name that group.
a) `$PY -m pytest -p no:cacheprovider tests/test_stage2_bundle.py -rs` -> `56 passed in 7.14s`, exit 0. Before the sets' arrays were copied: `55 passed, 1 skipped in 6.16s` (skipped: test_a_real_frozen_set).
b) `$PY -m designlab.stage2.bundle build --bundle-id standin --bodies 75-M-H-10,60-N-T-10 --sets "" --out <S>/standin` (S = /tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/wpc-cb), exit 0:
   `S2 BUNDLE standin: 2 bodies (60-N-T-10, 75-M-H-10), no set: 4 files in 2 parts, 1986560 bytes`
   `  models-60-N-T-10.tar: 1013760 bytes, 2 files, sha256 65ac70ea4cfc96abb46aaf8a277bb2a5e2339fe03ff68bd618400b8f54c81cae`
   `  models-75-M-H-10.tar: 972800 bytes, 2 files, sha256 2e9a55fcad2ff40b29ca1be706a6e2fddb99f6236d75997423de3264013f6917`
   `S2 BUNDLE standin: content sha256 eaeb1bcc4b76d246de44fa791ebb9d185391703065650504b73e85aa9da408a2 (manifest eaeb1bcc4b76); <S>/standin/MANIFEST.json written; parts in <S>/standin/parts`
   Three separate runs (two commits apart) gave the same two part hashes; the manifest's provenance read `dirty False`, `records_committed True`.
c) `... verify --manifest <S>/standin/MANIFEST.json --root . --parts <S>/standin/parts` -> `S2 BUNDLE standin: manifest eaeb1bcc4b76, 4 of 4 files match their sha256, 2 of 2 parts: OK`, exit 0.
d) `... unpack --manifest <S>/standin/MANIFEST.json --parts <S>/standin/parts --root <S>/unpacked` -> `S2 BUNDLE standin: manifest eaeb1bcc4b76, 4 files of 2 parts under <S>/unpacked: 4 written, 0 replaced, 0 already there: OK`, exit 0. Again: `0 written, 0 replaced, 4 already there: OK`. `verify --root <S>/unpacked`: `4 of 4 files match their sha256: OK`. `cmp` of the four unpacked files against the worktree's: same.
e) `git status --porcelain`: empty (the package is committed; the brief's "only your three files" is the diff stat of item 4).
f) `git check-ignore -v` in the worktree: `stage2/tb/work/x` (line 1 `work/`), `stage2/tb/bundle/x/parts/a.tar` and `a.tar.tmp` (line 14 `parts/`), `stage2/tb/bundle/x/.lock` (line 9), `stage2/tb/bundle/x/MANIFEST.json.tmp` (line 7 `*.tmp`) are ignored; `stage2/tb/bundle/x/MANIFEST.json`, `stage2/tb/runs/r/jobs/j/run.json`, `stage2/tb/gates/V6/b/g8a.json` are not (exit 1).
Extra, not asked for (files read and hashed only; scratch outputs deleted afterwards): the frozen set V0 of 75-M-H-10 with `--part-mb 5`: 22 files in 5 parts (set parts 4812800, 4505600, 4577280, 901120 bytes), verify `22 of 22 files ..., 5 of 5 parts: OK`; with 20: one set part of 14776320 bytes; build took 0.9 s. `--sets V6` -> `NOT BUILT: ... stage2/sets/V6/75-M-H-10: set.json is missing`, exit 1.

4. FINAL STATE
`git diff claude/mesozoic-trex-stage2-refine-4jffre --stat`:
 .../trex_form_study/designlab/stage2/bundle.py     |  904 +++++
 hardware/trex_form_study/stage2/tb/.gitignore      |   14 +
 .../trex_form_study/tests/test_stage2_bundle.py    | 1026 +++++
 3 files changed, 1944 insertions(+)
`git status --porcelain`: empty.
Left behind, all git-ignored or outside the repo: the copied inputs in the worktree (picks' models and arrays, the three sets' arrays); <S>/standin and <S>/unpacked.

5. FOR THE LEAD
Readings I chose (alternatives in brackets):
1. verify's `files_ok` and `parts_ok` are COUNTS (line 2 needs "<k> of <n>"); `parts_ok` is None without a parts directory. Because a count is truthy when some files fail, I added `ok` (bool), `parts`, `problems`, `bundle_id`, `content_sha256` (recomputed). [Alternative: booleans.] CN should test `ok`.
2. `unpack` has one extra keyword-only argument, `parts=None` (CLI `--only`), outside the frozen API: without it rule N4's second fallback (model parts alone) cannot unpack, since a missing part is refused. Default behaviour is the frozen one.
3. "MB" is 1,000,000 bytes and the limit is on the tar file's size, headers and padding included. [Alternative: MiB.]
4. Part index starts at 1 (`V6-<body>-1.tar`). The brief does not say.
5. `content.bodies` is sorted, so `--bodies A,B` and `B,A` give one content hash.
6. `models.<body>.record` is the model ledger JSON beside the XML (the second packed file), not the candidate record; `case` is the case directory's name.
7. Hashes "where the records hold them": the XML's from the candidate record's `xml_sha256` (also checked against `stage1.SHA256_PREFIX`); the ledger JSON's from the four committed inputs lists that `tools/restore_inputs.py` uses (hard-coded as `INPUT_LISTS`; a test pins them to the tool's and shows every registered body's two files are in exactly one). Where no list held a hash, the file's own would be used and `provenance.hash_sources` lists none.
8. Every (set, body) pair of a build must exist and be frozen; a missing pair is refused. [Alternative: skip.] If a look set exists for some bodies only, that needs separate bundles or an API change.
9. Added guard: where the tree is a git checkout, every record the build reads must be tracked and equal to HEAD, else refused ("commit it first"). The freeze's set.json must therefore be committed before `bundle build`. [Alternative: drop the guard; provenance already records `dirty`.]
10. A bundle id names one content: an existing MANIFEST.json with the same content is left untouched (first build's provenance kept); another content, or a foreign `.tar` in `<out>/parts`, is refused with nothing replaced.
11. A file that alone exceeds the part limit, or a body's model part over it, is refused; no part over the limit is ever written.
12. A relative path (API and CLI) is taken from the study directory, not the working directory (setio's rule). The acceptance's `--root .` is the same either way.
13. Provenance holds more than the listed keys: `dirty_paths`, `root`, `part_mb`, `part_limit_bytes`, `records` (sha256 of each record read), `records_committed`, `hash_sources`.
Disagreements between texts:
14. The design's work-package table asks for a stand-in of "the two picks' models and Stage 1 arrays"; the brief asks for models only (`--sets ""`). I followed the brief. Stage 1 run arrays cannot be packed at all under "never a file the freeze does not list" (the S1 view has no freeze).
15. Design section 4 says "an added provenance field", the brief "a changed provenance": both are tested.
16. The brief points to scratch copies of the design and rules; they are byte-identical to the committed ones in `stage2/design/` (`cmp`).
Not done / limits:
17. No bundle of V6 (not frozen). No upload, no `UPLOADS.json`. Not run on Colab or another Python; identical bytes were shown on Python 3.13.18 only, and `tarfile`'s header bytes can differ between Python versions (the version is in provenance). Colab only reads and hashes parts, so this matters only if parts are rebuilt elsewhere and compared by hash.
18. The full test suite was not run (forbidden); only my file.
19. The merge commit's message (item 1).
For the next packages:
- CN: `bundle.verify(manifest_path_or_doc, root, parts_dir=None)`; line 2 from `bundle_id`, `content_sha256[:12]`, `files_ok`, `files`, `ok`, `first_problem`. `bundle.unpack(manifest, parts_dir, root, *, parts=None)` raises `bundle.BundleError` (an `s2io.S2Error`). Neither checks that `content.bundle_id` equals BUNDLE_ID: compare it. Importing `bundle` imports numpy, so `nb.py` must import it lazily or use the CLI (exit 0 OK, 1 failed or refused; lines start `S2 BUNDLE`).
- All: under `stage2/tb/` the patterns `checkpoints/`, `logs/`, `partial/`, `traces/`, `parts/`, `work/` ignore whole directories, so `checkpoint_<n>.json` under `checkpoints/` is ignored too.

6. COMMITS (oldest first)
caaf14c70c4f6a88bb76fe246032bc98da8cb11e  Track B: the ignore file of stage2/tb
085b69fb3e697fcbd3cbe2573b164bb5730767a1  Track B: the Drive bundle tool (build, verify, unpack)
7999abfe247d8775c8321e177bce4d2a258ad30e  Track B: tests of the bundle tool and of the ignore file
55564c7e10e450154c9d7dd4bf88b69f9ded6889  Bundle: unpack refuses a blocked place before it writes
7f01dda0e444c24b57a362a2060663fe879def45  Bundle: counts in the printed lines read as singular or plural
5aaf0211162e3e0ac8169d8285fce339ab36ed30  Merge branch 'claude/mesozoic-trex-stage2-refine-4jffre' into wt/wpc-cb (default message, no attribution lines: see item 1)
The first five end with the two attribution lines (checked by script).
