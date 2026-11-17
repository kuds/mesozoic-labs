Verdict: merge

**Verdict: merge.** Fast-forward `wt/wpc-ct` at bb1d1f99ed3e17d0a3c95eced94678f8b39d99d0 onto the working branch. Both fixes are in, and each has a test that I saw fail with the fix taken out. My read of the look-set path found no other collision. My own tiny end-to-end run reached G8a, the baseline check, 8 PPO updates and 2 monitor checks, and ended done. Simulation only. I changed no tracked file and committed nothing.

**Merge readiness**
- The working branch `claude/mesozoic-trex-stage2-refine-4jffre` is at 5e1b949f41f9b4334f97856b4f60141c1c70731a, and it is an ancestor of the tip (`ancestor-ok`). So the merge is a pure fast-forward; I made no merge myself.
- The diff against the working branch touches only `train2.py` (+16/-2) and `tests/test_stage2_train2.py` (+67). `bar.py` and `moves.py` have no diff.
- Commits 8f6adb64 and bb1d1f99 each end with exactly the two required lines. Neither names a model anywhere else.
- The worktree is clean, and its ignored-file list is the same before and after my runs. The main checkout has 0 porcelain lines and HEAD 5e1b949f.
- Bit-identity: `45719d030359ec7d5d3914af3814d837fdc05d77652ba45c64792b511027528d c0027992de21067525cc4489b74acb157feb89f3a4c7618580addc2b341be1c1`.

**Fix 1: `_g8a` (ruling 43)**
- `_g8a` renames the form's `at` to `form_at` and stamps the record's own `at`.
- `g8a.bar_form` returns these keys: rule, body, set, set_freeze, cut_s, episodes, identical, of, not_run, passes, sources, key_context, versions, blas_threads, commit/dirty/dirty_paths and at. With `form` added, none of them collides with schema, job, run_uuid, tiny, session or sha256.
- Nothing else reads `g8a.json`.
- To check that the test fails without the fix, I wrote a pytest plugin (`scripts/prefix_plugin.py`). It compiles the working branch's own `_g8a` source, taken with `git show`, into train2's namespace. Nothing on disk changes.
  - With the old `_g8a`, the test fails: `E   TypeError: dict() got multiple values for keyword argument 'at'` (train2.py:813 in go) → `1 failed, 1 passed`.
  - Unpatched, both new tests pass: `2 passed in 10.69s`. The look-set test ran the real `bar_form`; it was not skipped.

**Fix 2: the read for other spreads**
- I read every `**` in `train2.py` and each `monitor2` spread the trainer reaches: `run`, `baseline`, `_also`, `no_record`, `decide`, `g8b` and `s2_10`. My result matches the builder's list. Every one spreads fixed key sets that do not overlap with the keys beside it, or merges a mapping, except one.
- That one is `_fresh`'s `PPO(env=, seed=, device=, verbose=, **kw)`. `kw` takes the caller's `ppo`; `policy.model_kwargs` alone names none of those four.
- The fix (`PPO_OWN_ARGS`, refused in `_Run.__init__` before anything is written) is in with a test. With `PPO_OWN_ARGS` emptied, the test fails: `E   TypeError: stable_baselines3.ppo.ppo.PPO() got multiple values for keyword argument 'seed'` (train2.py:1170 in _fresh).
- Outside the scope, in `g8a.py`, the spreads `**s2io.git_state(), at=`, `**c` and `**why` (bar's `missing`: reference/numbers…) also use disjoint keys.

**The tiny end-to-end run (scratch, not a record, no study set)**
- **Isolation.** I used no path patching. I extracted bb1d1f99 with `git archive`, taking `hardware/trex_form_study`, `environments`, `configs`, `conftest.py` and `pyproject.toml`, and copied in the four ignored 75-M-H-10 input directories. In that copy I deleted the real `stage2/sets/V6` and `stage2/tb/gates/V6`. I then wrote a synthetic V6 at the copy's own `setio.ROOT` (`make_look`, `stand_in_walks`; manifest f36cf60cd28f…987c3; the real V6 is 2eeaebe8…8def) and a passed G9-ref with `passed_g9_ref`.
- **Commands.**
  - The baseline was made as V8 makes it: `monitor2.baseline('75-M-H-10','V6',dir,tiny=True)` → `valid True N 0 S 15.126666666668307 outcomes {'read': 1, 'fell': 5, ...} episodes 6`.
  - The run went through the CLI: `train2 run --mode R5 --body 75-M-H-10 --seed 42 --set V6 --tiny --workers 2 --expect-baseline <that file>`. It used the real `_monitor_cmd` and worker command, and gave `train exit 0`.
- **Key log lines:**
  - `R5-75-M-H-10-s42: G8a's short form on set V6`
  - `R5-75-M-H-10-s42: building 8 envs in 2 workers`
  - `baseline R0L@10ms: N 0, S 15.127 s (6 episodes, 12 s)`
  - `R5-75-M-H-10-s42: R0L's baseline equals the committed one (...check_fix/e2e_out/baseline/r0l.json): rule M3`
  - `R5-75-M-H-10-s42: first rate: 522 env steps/s (8 envs, 2 workers, 4 vCPUs)`
  - `R5-75-M-H-10-s42: check at 2048 steps: N 0 (N* 0), S 15.019 s (S* 15.127 s): no gain`
  - `R5-75-M-H-10-s42: check at 4096 steps: N 0 (N* 0), S 14.688 s (S* 15.127 s): no gain`
  - `R5-75-M-H-10-s42: done at 4096 steps`
- **What `run.json` holds:**
  - Status done, 4096 steps, halt and stopped null.
  - `gates.g8a` passes 2 of 2, form "bar".
  - `g8a.json` has one `at` (2026-11-17T20:02:09Z) and `form_at` 2026-11-17T20:02:09Z. Its set_freeze is the synthetic manifest, and both walk/v0.4 episodes compared 1000 of 1000 steps and are identical.
  - `baseline_expected`: equal True, 6 episodes, digests_compared True, differ [].
  - 8 updates, each running 10 of 10 minibatch steps; lr 9.13e-06 down to 3e-06; filter 10 Hz.
  - Both checks: one attempt each, exit 0, a record, decided. G8b at 2048: up 1 of 1, pass.
  - The session's commit is None, because the copy has no git. The builder's run in the worktree recorded commit bb1d1f99.
- **No further fault in the trainer package.**
  - My first attempt failed only because my copy lacked `configs/species_manifest.toml`. That was my harness's fault, and the log is kept as `logs/e2e_attempt1_no_configs.log`.
  - The numbers match the builder's run (S 15.127, 15.019, 14.688).

**Tests on bb1d1f99**
- `tests/test_stage2_train2.py tests/test_stage2_monitor2.py` gave exit 0, with 88 `.` and 3 `s`.
- The repo's `addopts -q` plus my `-q` hid the final count line.
- The skips are `tests/test_stage2_train2.py:2500/2526/2545: real env on the S1 view; set STAGE2_SLOW=1`. I did not run those slow tests.

**Hashes**
- `train2.py` b7491e12dd87c7ffa75f03fe29908c6e11e7e33c5440ab58b4bd037109959bb6
- `monitor2.py` f7f1b180533df8008dd6c82647bab37beeb454f00b2c02dd9a2523b1a1cd8efa
- `clip2.py` 8e4617a8d84f9a776f218ea6f9c21036375ff6a06c2a926790233e57e82f6068
- `test_stage2_train2.py` 5f049d5d5354895bfd8915f0663327d16a5c2d4c907afca91ed9d88858caa9a9
- `test_stage2_monitor2.py` 7abf7bf9245b01e9ee3b7038c8cb04d5eacffc8197eadd7a6df0c028e70694e3
- These all match the builder's hashes, and the copy's `train2.py` and `monitor2.py` are identical to these.
- Scratch `run.json` 3988f1fcb8cce1aac9ffef10018221679498b7070ed0214db6c25e05b2df12c3
- Scratch `g8a.json` 168449ae66c6e9beb373ea840f6fbc637854304dc698816d95631595894d711c
- Baseline 6ddd7434a04701904e53fe05bdb3e6da58752c20be651f8ff885e46db54a6ba7

**For you to decide: the builder's rule X1 gap (report, not fix)**
- In `train2.py`, nothing checks G9-ref before `_g8a`. The only refusal ("the look set's G9-ref has not passed (rule X1)") comes from the monitor child at the baseline, after G8a's closed-loop episodes have run.
- `g8a.py`'s docstring leaves X1 to the caller.
- I confirmed this from the code only. I did not repeat the builder's no-G9-ref probe.
- It does not touch S2-4, because V6/75-M-H-10 has a passed G9-ref.

**Not done:** the STAGE2_SLOW tests, the X1 probe, and a run inside the worktree itself (mine ran in the isolated copy).

Everything is under `/tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/s24b/check_fix/`. The pytest temp dirs and the 453M copy are deleted.
- Scripts: `scripts/prefix_plugin.py`, `scripts/setup_synth.py`, `scripts/e2e.sh`
- Logs: `logs/tests_head.log`, `logs/reverts.log`, `logs/setup.log`, `logs/e2e.log` (sha256 7861bcec…b3)
- The scratch run: `e2e_out/`
