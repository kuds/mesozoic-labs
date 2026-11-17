# check:CT3 (roundC, 17 Nov, second attempt): the agent's final report

Working paper. Nothing here is a rule.

VERDICT: merge as it stands. I found no blocker and no major defect. The four findings below are minor or nits, for the next change to these files. The working branch moved twice while I reviewed (f64094e2, the rows package CW merged; then 7df517c1, docs only). The merge with the current tip is clean, touches only the package's files, and its tests pass (below).

FINDINGS, MOST SERIOUS FIRST

1. Minor. One kind of checkpoint record with the wrong shape is still no refusal.
   - Where: designlab/stage2/monitor2.py:763-770 (`_checkpoint`) checks the shape of `files` only.
   - What happens: take a record with the right schema, steps, file names, sizes, sha256 and self-hash, and a model zip that holds `data` and `policy.pth`, but no `layout`. It gets past `_checkpoint` and `s2load._inference_problem`. Then `s2load._read_fingerprint` (s2load.py:651, `record["layout"]`) raises a KeyError inside `lookbar.rows`. `_row_refusal` passes it on, and `monitor2.main` does not catch it: a traceback, exit 1, a counted attempt.
   - Evidence: my probe on the merged tree with the real s2rows and lookbar printed `[D record without layout] raised out of monitor2.main: KeyError: KeyError('layout')`.
   - The same applies, by reading only, to a `fingerprint` entry that is a non-empty non-object (s2load.py:665 `.get` on it: AttributeError).
   - A cut write cannot produce such a record, since it must hash to itself. So this is far from fail-open.
   - Fix: in `_checkpoint`, also require `isinstance(rec.get("layout"), dict)` and `isinstance(rec.get("fingerprint"), dict)` (and optionally `rec.get("steps") == steps`), and raise `MonitorRefusal("checkpoint", ...)`. Or harden s2load, which is not CT's file.

2. Minor, for the lead. I confirm the builder's lead item 1 against the real row; the halt does not say how to fix it.
   - The real `s2rows.source_check` compares every session's recorded hashes (s2rows.py:154-172). The trainer's resume compares only the last session's (train2.py:776-790).
   - So a run resumed once on moved sources needs `--allow-source-change` again at every later session. Otherwise its next check halts with exit 5.
   - Probe C1: the real row refuses on session 0's stale hashes, `exit 8; note kind 'row'`.
   - The halt quotes the row's text "STAGE2_ALLOW_SOURCE_CHANGE=1 runs it flagged", not the trainer's parameter. The notebook (nb.py:2581 on wt/wpc-cn) takes only `retry_check` from a halt's text, so the lead must know to write `allow_source_change` into LEAD.json.
   - It halts, so it does not fail open.
   - Options: the builder's alternative B (also pass the variable when run.json `flags.source_change` is set); or keep the literal reading and have the halt name `--allow-source-change` when the refusal kind is "row" and the reason cites rule K9.

3. Nit, test gap. No test tells `after_halt` from the session's `--retry-check` in the halt text (train2.py:1328-1330).
   - My mutation of `if after_halt` to `if self.retry_check` survived (`-k "refus or retry"`: 9 passed).
   - The two differ only in a `--retry-check` session whose refused attempt was a fresh one. There the text would wrongly name the parameter, and the notebook would set the remedy retry_check. The harm is small.
   - Fix: one assertion in a refusal test run with `retry_check=True` and no attempts left open.

4. Nit, stale text after CW was merged at f64094e2.
   - "s2rows is not merged yet" at monitor2.py:95; tests/test_stage2_monitor2.py:28 ("the rows package is not merged") and :397 ("neither is merged here").
   - `_stand_in` and its new refusal wrapping can no longer be reached from `_episodes`; only the direct test reaches them.

WHAT I CHECKED AND FOUND RIGHT
- The rulings are met: 10 and 11 of 17 Nov, and ruling 3 of the sixth list (read from the files in the main checkout).
- (a) `monitor_env` gives "1" exactly when the session has `--allow-source-change`, and drops a value from the trainer's own environment. Its only use is in `_child` (train2.py:1154), which starts the baseline and every check.
  - `s2rows.SOURCE_CHANGE_ENV == train2.ALLOW_SOURCE_ENV` is True on the merged tree.
  - The real row reads the variable inside `build` (s2rows.py:260). Probe C2, with the variable at "1", passes K9 and refuses later, at set_check ("records no input file").
  - The notebook passes `--allow-source-change` as a flag (nb.py:2424-2425 on wt/wpc-cn), so dropping the inherited variable does not break it.
  - On a fresh run the flag is harmless: the monitor's own sources check (train2 SOURCE_FILES plus the plant and panel hashes) covers everything the row compares within a session.
- (b) The narrow rule is an isinstance check against the class in sys.modules, so a subclass also counts. It wraps only `s2rows.configure` and `lookbar.rows`. `lookbar.episodes` is outside it, and a refusal at a further delay becomes a MonitorError, exit 1.
  - With the real modules, the factory builds the row inside `lookbar.rows` (`panels.tracker_rows`, then `s2rows._factory`, then `build`), before any episode.
  - Probe A: a real RowError at build gives `exit 8; note kind 'row'; episodes called 0`.
  - Probe B: a real CheckpointError from `load_checkpoint` at the row's load gives `exit 8; note kind 'checkpoint'; episodes called 0`.
  - The probe replaced `lookbar.episodes` with a trap: 0 calls in all.
- (c) A run.json that is missing, unreadable or not a JSON object is a refusal of kind "run" at a check and at the baseline. `s2-10` keeps exit 1. The shape check rejects bool `bytes` (`type(...) is int`).
- (d) `after_halt` is right: under `--retry-check` the two attempts never closed stay open, so only a `--retry-check` resume runs the check again. The kill test pins this.
- (e) The kill test SIGKILLs the trainer between the monitor's exit 8 and the attempt's closing, twice. It ends bit for bit the uninterrupted run (checkpoints 1024 and 1536, updates.jsonl).
- No failure path I found fails open: every new refusal halts with exit 5 and is never read as a pass.
- Diff against the working branch: only monitor2.py, train2.py, test_stage2_monitor2.py and test_stage2_train2.py. clip2.py is unchanged.
- `git status --porcelain` is empty in the worktree and in the main checkout. The merge commit 6999e818 has the same tree as fc505b0d. CT's files were identical on fc505b0d and 2fca27bf.
- Merge readiness: `git merge-tree --write-tree claude/mesozoic-trex-stage2-refine-4jffre HEAD` exits 0 at both tips.
  - At f64094e2: tree 84532051.
  - At 7df517c1: tree 3667af58.
  - Each differs from its tip only in the four package files. 84532051 and 3667af58 differ only in five docs files.
  - The rulings added at 7df517c1 say nothing about CT.
- Record-key hashes (in the worktree): `45719d030359ec7d5d3914af3814d837fdc05d77652ba45c64792b511027528d c0027992de21067525cc4489b74acb157feb89f3a4c7618580addc2b341be1c1`, the registered pair.
- Nothing ran that should not have: no training, no episode, no look set. The worktree's ignored `designlab/stage2/__pycache__` dates from 16 Nov 19:17, before this round.
- The builder's mutation log exists and says "15 of 15 mutations caught". My own 8 mutations on the merged tree: 7 caught (row refusal by class name only; `lookbar.episodes` inside the refusal block; bool `bytes`; an unreadable run.json as MonitorError; the baseline not refusing on run.json; the flag always passed; the inherited value passed on). The survivor is finding 3.

EXACT TEST SUMMARY LINES (my runs, standing environment, through tools/lowprio.sh)
- Worktree at eee35062:
  - tests/test_stage2_monitor2.py: "26 passed in 29.36s" (exit 0)
  - tests/test_stage2_train2.py: "50 passed, 3 skipped, 2 warnings in 211.01s (0:03:31)" (exit 0). The 3 skips are the `slow` marker (STAGE2_SLOW).
- Merged tree 84532051 (f64094e2 plus CT, a git-archive export with the ignored inputs linked; same code as the merge with 7df517c1):
  - tests/test_stage2_monitor2.py: "25 passed, 1 skipped in 31.40s" (exit 0). The skip is the older test_without_lookbar_only_the_stage_1_view_has_a_stand_in: "lookbar is merged: the stand-in is no longer reached".
  - tests/test_stage2_train2.py: "50 passed, 3 skipped, 2 warnings in 221.69s (0:03:41)" (exit 0). My first run there gave "9 failed, 22 passed, 3 skipped, 2 warnings, 19 errors in 79.80s (0:01:19)". The cause was my export, which lacked configs/ (FileNotFoundError on configs/species_manifest.toml). After I added configs/, it gave the line above.
- ruff on the five package files: "All checks passed!"

COMMITS TO BRING IN
- Six commits are not in the working branch, oldest first: 57b5bf36, 2fca27bf, 6999e818 (the merge), 695e8429, 9c348ca7, eee35062.
- Every one ends with two attribution lines:
  - 57b5bf36 and 2fca27bf carry the earlier model's lines, right for their time. Their content is already on the working branch (0ad2072d, bf9bc434), so they change no file.
  - The other four end with the given two lines.
- No commit message names a model anywhere else.
- So: merge the tip eee35062 into claude/mesozoic-trex-stage2-refine-4jffre, with the message written to a file, `-F`, and the two lines at its end.

WHAT I DID NOT CHECK
- No STAGE2_SLOW tests and no real env.
- No real fingerprint mismatch on a real checkpoint: the CheckpointError reached the row through the record-schema path, not the fingerprint branch.
- No end-to-end trainer, real monitor child and real row run. The variable's passing is covered by the stub-child test plus my in-process probe C1/C2 (`s2load.load_checkpoint` patched for C).
- No Colab or Drive behaviour.
- I did not run the notebook's tests; I only read its halt parsing on wt/wpc-cn.
- I did not re-run the builder's mutation script.

FILES (all in /tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/rerun17/check_ct3_17nov/)
- Scripts: probe_real_rows.py, mutate_check.py
- Logs: probe_real_rows.log, mutate_check.log, wt_monitor2.log, wt_train2.log, merged_monitor2.log, merged_train2.log
- The merged export was deleted after use.
