# review:CB (roundC, wave 2a): the agent's final report, 16 Nov 2026

Working paper. Nothing here is a rule.

REVIEW OF WORK PACKAGE CB (Drive bundle tool), branch wt/wpc-cb at 5aaf021. Simulation tooling only; nothing uploaded, no network call, no episode run by me. I changed no tracked file and committed nothing.

VERDICT: merge as it stands, with one condition on how: merge commit 7f01dda (the last package commit), not the branch tip 5aaf021. No blocker and no wrong result found. One design conflict needs the lead's ruling before package CN (finding 1). The rest are minor follow-ups.

FINDINGS, most serious first

1. MAJOR, for the lead (a conflict between design texts, not a builder error). `stage2/tb/.gitignore` line 13 `checkpoints/` hides the checkpoint records the design wants committed.
   - The brief and D22 give the 14 lines verbatim, and the builder wrote exactly those.
   - But the design's layout (ROUND_C_DESIGN.md line 1637) lists `jobs/<job>/{..., checkpoints/checkpoint_*.json}` among the committed records, and WPC_CN.md item 8 has `collect` copy "of `checkpoints/` only `checkpoint_<n>.json`".
   - Evidence, in the worktree: `git check-ignore -v stage2/tb/runs/r/jobs/j/checkpoints/checkpoint_1.json` gives `.gitignore:13:checkpoints/`.
   - Evidence, in a scratch repo with this ignore file: `git add stage2/tb` on a collected job staged `run.json`, `updates.jsonl`, `gates/g8a.json`, `monitor/m_1.json` and silently left out `checkpoints/checkpoint_1000000.json`. Adding it by path fails with "paths are ignored".
   - The package's test pins this side: `tests/test_stage2_bundle.py:174` lists `runs/r/jobs/j/checkpoints/checkpoint_1.json` under IGNORED.
   - Options: (a) drop the `checkpoints/` line; (b) keep it and have `nb collect` put the records in a directory with another name; (c) keep it and commit with `git add -f`.
   - I would choose (a). The three binaries are already covered by `*.zip`, `*.pkl`, `*.npz`, and partial copies by `*.part` and `*.tmp`; live job directories sit under `work/`. It needs D22's text amended, one line removed, and the test's path moved to NOT_IGNORED. Not needed before this merge; needed before CN.

2. MINOR (rule). The tip 5aaf021 is a merge commit with git's default message and without the two attribution lines, as the builder reported.
   - Options: merge 7f01dda instead of the tip; amend the tip (a history rewrite); accept it.
   - I would merge 7f01dda. `git merge-tree --write-tree claude/mesozoic-trex-stage2-refine-4jffre 7f01dda` and the same with HEAD both give tree 5c26c3893f2211c6eba18e5ba924592689dbfcf5, exit 0. The unattributed commit then never enters the working branch and nothing is rewritten. The branch is on no remote.

3. MINOR (interface to CN). WPC_CN.md line 69 says `nb.py` reaches `bundle` only by subprocess, so CN sees the CLI's text, not the result dict.
   - The `verify` line prints only the first 12 hex of the content hash; SESSION's `bundle.content_sha256` needs all 64.
   - The wording has already changed once (commit 7f01dda, singular and plural).
   - Fix: add `--json` to `verify` and `unpack` (one JSON line of the returned dict), or have CN call `python -c "... bundle.verify(...)"` in its subprocess. I would add `--json` in CB.

4. MINOR (test gaps). I put 33 single faults into a scratch copy of `bundle.py`; 28 made the test file fail, 5 did not:
   - M16, unpack leaves a wrong file of the same size in place (line 783 reduced to a size check). The only "wrong file" test uses a 5-byte file. The behaviour itself is right: my probe flipped one byte and it was replaced. Add a same-size case to `test_unpack_gives_the_files_and_twice_is_a_no_op`.
   - M34, `_names` no longer sorts. The builder's point 5 (bodies in either order give one hash) has no test. It does hold: reversed bodies gave the same hash on the synthetic tree and on the real picks. Add one assertion.
   - M24 and M27, the re-checks against a file changing mid-run (lines 799 and 499). They guard races only; a test would need a monkeypatch.
   - M25, PAX instead of USTAR: byte-identical for these names (58 characters at most in a real set), so an equivalent mutant.

5. MINOR (failure paths that end in a traceback instead of a refusal; all fail closed with a non-zero exit, nothing written).
   - `_set`, lines 263-270: a `set.json` that is not JSON raises JSONDecodeError; a freeze without `index` KeyError; a list for `set.json` or `refs.json` AttributeError. The CLI catches only `S2Error`. Fix: catch these and append to `bad`.
   - `unpack`, line 766: `_hash(path)` is outside any try. With ENOTCONN simulated on a part, unpack raised OSError, not BundleError.
   - `verify`, line 679: `path.is_file()` re-raises errnos other than ENOENT and its kin. With ENOTCONN simulated on a part's stat, verify raised, against "never raises". This matters only if parts are read straight from the mount; the design copies them to local disk first.
   - A manifest file nested 100,000 deep raised RecursionError in `verify` (pathological).

6. MINOR. `unpack --only ""` prints "0 files of 0 parts ... OK", exit 0 (lines 750 and 887). Refuse an empty selection.

7. MINOR. Temporary parts of a killed build are removed only if the next build uses the same part names (line 539). After a killed build, a build with other names left `SX-T-A-3.tar.tmp` and `SX-T-B-4.tar.tmp` in `parts/`. They are git-ignored and never read. Fix: `s2io.sweep_tmp(parts_dir)` under the lock.

8. MINOR. Where git cannot tell (no checkout, no git, a "dubious ownership" refusal), the committed-records guard becomes a printed note and `provenance.records_committed: null` (lines 341-348, 491). For the real bundle the lead should check the manifest reads `records_committed: true` with a non-null `commit` before committing it; or `build` could refuse when `root` is the study itself.

9. MINOR, optional.
   - `manifest_problems` checks `files` and `parts` only. In my fuzz, a manifest with junk in `bodies`, `models` or `sets` and a refitted hash verified OK. The defence is that CN compares the content hash and bundle id with the quoted ones (the builder says so). CN could also compare `content.sets[set][body].manifest_sha256` with the clone's latest freeze.
   - `provenance.commit` is the tool's checkout, not `root`'s, when `root` is given.
   - `unpack` takes no lock; the `verify` the design runs after it is the safeguard.

WHAT I CHECKED AND FOUND RIGHT
- Brief and design: the 14 ignore lines byte for byte; `build`, `content_sha256`, `verify`, `unpack` and the three CLI commands; the tar rules; schema `s2-bundle-2` with the content block of 3.10 and provenance outside the hash; models from the registry with no plant import and no rebuild; latest freeze only, no look twins, no unlisted file; split and part names. Every row of the design's "Bundle" test line and of the brief's test list has a test.
- Acceptance, rerun in my scratch directory: build, verify, unpack, unpack again and verify of the unpacked root all exit 0, with the builder's exact hashes (parts 65ac70ea..., 2e9a55fc...; content eaeb1bcc4b76...). A second build under umask 077, another cwd, hash seed and time zone, with the bodies reversed, gave byte-identical parts (`cmp`) and the same content hash. The unpacked files equal the worktree's; GNU tar lists and extracts a part correctly.
- Other Pythons, which the builder did not test: under 3.11.18 and 3.12.3 here (with a stub numpy, since those have none) `_write_part` reproduced all 10 parts of a 3.13-built bundle with identical sha256. Verify and unpack worked and a tampered part was refused.
- A stand-in fresh clone (`git archive` of `designlab` and `stage2/sets`, no arrays): a bundle of the frozen set V0 (22 files, set part 14,776,320 bytes, 1.1 s) unpacked into it. `verify` gave 22 of 22 and `setio.read_inputs(require_frozen=True)` accepted all 20 arrays without the look twins. Files were hashed only.
- Builder's extras confirmed: `--part-mb 5` gives 5 parts (4812800, 4505600, 4577280, 901120 bytes); `--sets V6` is refused ("set.json is missing"), nothing written.
- Kill and resume (`os._exit` at five points): a build killed after three temporary parts, before the manifest, or mid-manifest is completed by the same build again, with the stale lock taken over and parts identical to the reference. Another content after such a kill is refused where finished parts remain. An unpack killed mid-way is completed by a second unpack, no `.tmp` left, verify OK. A second builder on the same directory is refused by the lock.
- Committed-records guard with the study as a subdirectory of the repo (as in the real one): a changed, a staged-only, an untracked and a git-ignored record are each refused; a foreign `GIT_DIR` in the environment also refuses.
- Fuzz: 3000 damaged manifests; `verify` and `unpack` raised nothing but `BundleError`, and nothing wrong was written. Crafted parts with a refitted manifest (PAX path override, GNU long name with `..`, device, fifo, sparse, trailing slash, `./` name) were all refused with nothing written.
- Importing `bundle` loads only `designlab`, `stage1`, `stage2`, `s2io`: no mujoco, torch, scipy, plant or setio.
- Rules:
  - Hashes: the two record-key hashes are the registered pair at the worktree's HEAD and on the merged tree (`45719d03...528d c0027992...e1c1`); `bundle.py` is in no key.
  - Diff: `git diff claude/mesozoic-trex-stage2-refine-4jffre...HEAD --stat` shows the three owned files, 1944 insertions. The two-dot diff also shows three other files (servo-model notes; later also `checkpoint/STATUS.md`), only because the working branch moved after the builder's last merge.
  - Commits: the five package commits touch only owned files and end with the two lines; no model name elsewhere in messages or files.
  - Leftovers: the worktree's ignored files are only the restored inputs; `stage1/models` was not rebuilt; both checkouts have an empty `git status --porcelain`.
- Merge readiness: the working branch moved to 82ba1ad during the review. The merge is still clean and brings exactly the three files. No other local branch has a file under `stage2/tb`.
- The three Track B rulings files contain nothing about the bundle or the ignore file.

TEST SUMMARY LINES (mine)
- Worktree: `56 passed in 6.78s`, exit 0 (28 functions, 56 collected).
- Scratch copy without the git-ignored inputs: `53 passed, 3 skipped in 5.71s`.
- `ruff check` on both files: `All checks passed!`

THE BUILDER'S POINTS FOR THE LEAD
1. `files_ok` and `parts_ok` as counts plus `ok`: keep. Line 2 needs k of n, and CN must gate on `ok` or the exit code. The alternative, booleans plus separate counts, removes the truthy-count trap but changes nothing else.
2. `unpack(..., parts=None)` and `--only`: accept. N4's second fallback needs it and the default is the frozen behaviour.
3. MB as 1,000,000 bytes, on the tar's size: accept; it is the stricter reading.
4. Part index from 1: accept.
5. Sorted bodies: accept; add the test (finding 4).
6. `models.<body>.record` as the ledger JSON beside the XML: accept; it matches "the two model files" of 3.10.
7. Hash sources (candidate record plus prefix; the four inputs lists): accept; a test pins the lists to `restore_inputs.py`.
8. A missing (set, body) pair is refused: accept. A silent skip would fail open; use separate bundles if a set exists for some bodies only.
9. Committed-records guard: keep. It means the freeze's `set.json` is committed before `bundle build`; D24's C1 still holds both. See finding 8.
10. One bundle id, one content: accept.
11. A single file over the limit is refused: accept. V0's largest array is 1.38 MB, so the 5 MB fallback has room there; V6 is unknown until frozen.
12. Relative paths from the study directory: accept (setio's rule). I confirmed `--manifest MANIFEST.json` from another cwd looks in the study directory. CN runs with cwd there.
13. Extra provenance keys: accept; they are outside the hash.
14. Stand-in of models only: the brief wins. Stage 1 arrays have no freeze and cannot be packed; a frozen set (V0) was tried instead.
15. and 16.: nothing to decide.
17. Not run on Colab: accept; my 3.11 and 3.12 check narrows it. Build the real bundle with the study venv.
18. Full suite not run: correct under the rules.
19. Merge commit message: finding 2.

NOT CHECKED
- No run on Colab or on a mounted Drive; the other-Python check used a stub numpy and could not build a set there (no mujoco).
- No bundle of V6 (not frozen); no upload, no `UPLOADS.json`.
- The full test suite (forbidden); `test_stage2_bundle.py` only.
- What the builder ran beyond what the files and commits show.
- Rules N3 and N4 only as far as the tool goes; the notebook's side is CN's.

My scripts and outputs are in /tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/review_cb_wave2a/ (mutate.py, probe1.py to probe4.py).
