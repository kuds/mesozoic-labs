Ruling 43 is done: the `_g8a` fix is in with a test, the read for `**` spreads turned up one more collision (fixed with a test), and the tiny end-to-end run on a synthetic look set completed. That run hit no further fault. A separate probe found one rule X1 gap in the trainer, which I am reporting for you rather than fixing. Simulation only; nothing pushed, nothing amended, `bar.py` and `moves.py` not touched, main checkout clean (HEAD 5e1b949f).

**Merge**
- Worktree `/home/user/mesozoic-wt/wpc-ct` (branch `wt/wpc-ct`) was behind, so `git merge --ff-only claude/mesozoic-trex-stage2-refine-4jffre` fast-forwarded it from 3e99223a to 5e1b949f. There is no merge commit.

**Fix 1, `_g8a` (commit 8f6adb64)**
- `train2._g8a` now renames the form's `at` to `form_at` and stamps the record's own `at`. The docstring says so.
- Test `test_g8a_on_a_look_set_keeps_the_forms_own_time_as_form_at`:
  - It builds a synthetic look set V6 in a temp dir (`test_stage2_r5lib.make_look`) and runs the trainer's own `_g8a_short` (bar form, `walk/v0.4`, 2 s cut) in a tiny run with gates; the env and monitor are stubs.
  - It checks `form` "bar", `form_at` equal to the form's own `at`, the remaining keys equal to the form, the sha256, `run.json` `gates.g8a`, and status done.
  - Without the fix it fails at `train2.py:920`: `E   TypeError: dict() got multiple values for keyword argument 'at'`. With the fix it passes.

**The read for other `**` spreads**
- **Fixed (commit bb1d1f99):** `_fresh` builds `PPO(env=..., seed=..., device="cpu", verbose=0, **kw)`, and `kw` takes the caller's `ppo` overrides. The CLI never passes `ppo`, but a Python caller could.
  - An override naming `env`, `seed`, `device` or `verbose` is now refused when the run is set up, before anything is written (`PPO_OWN_ARGS`).
  - Test `test_ppo_overrides_that_name_what_the_trainer_sets_itself_are_refused`. Without the fix: `E   TypeError: stable_baselines3.ppo.ppo.PPO() got multiple values for keyword argument 'seed'`.
- **Cannot collide in `train2.py`:** each of these spreads a fixed set of keys, or keys it excludes, or merges a mapping (which never raises):
  - `_g8a_short` `dict(form="bar", **doc)`: `bar_form` returns fixed keys with no `form`.
  - `_begin` `**git_state`: commit, dirty, dirty_paths only.
  - `session.update(**env_described)`: env, other_envs only.
  - `_rollback` `**derived`: excludes `at_steps`.
  - The `att.update` calls and the attempt dict: the literal keys refused, refusal, after_halt.
  - `**same`: `baseline_same`'s fixed keys.
  - `final_check` `**extra`: fix_not_applied, would_have_stopped.
  - `_rate_record`: only `first_rollout` and `first_three`; `session` is an int and filtered out.
  - `baseline_entry`: `SUMMARY_KEYS`.
  - The probes' `_noise_summary`: episodes, survived, mean_survival_s.
  - `_probe_cell` and `preflight` `PPO(**kw)`: no caller keys.
  - `dict(doc, **_stamp)` and `dict(have, **out_bodies)`: mapping merges.
- **Cannot collide in `monitor2.py`, where the trainer calls it:** `run`, `baseline`, `_also`, `no_record`, `decide`, `g8b` and `s2_10` spread only `summarize` or `_summary`, which are fixed literal key sets disjoint from each record's own keys.
- A related observation, not a fault: the trainer's G8a short form runs outside `lookbar.look_set_placement`. It plays only `walk/v0.4`, which placement does not change.

**The tiny end-to-end run (scratch, not a record, no study set)**
- **Harness:** `s24b_e2e.py` (below).
  - It points every process at `$S24B_ROOT`: it rewrites the 14 function defaults that were the study's `setio.ROOT` and sets `setio.ROOT` and `gates2.GATES`. The trainer, the monitor child and the 2 pool workers run through the driver, using the `_monitor_cmd` seam and `WORKER_CMD`.
  - Every process printed `unpatched: []`. The worktree had no new files afterwards, ignored ones included.
- **Setup:** synthetic V6 (manifest f36cf60c…87c3) and a passed G9-ref written with the lookbar fixture `passed_g9_ref`.
- **Baseline made the same way:** `monitor2.baseline(BODY, "V6", out, tiny=True)` gave valid, 6 episodes (read 1, fell 5), S 15.127, every episode with a state digest. File sha256 d73dbd2e…86d5.
- **Run:** `train2.train("R5", 75-M-H-10, 42, V6, tiny=True, workers=2, expect_baseline=…)`, 19:38:00–19:39:46Z, exit 0. It got all the way through. Key lines:
  - `R5-75-M-H-10-s42: G8a's short form on set V6`
  - `baseline R0L@10ms: N 0, S 15.127 s (6 episodes, 14 s)`
  - `R5-75-M-H-10-s42: R0L's baseline equals the committed one (…/e2e_root/baseline/r0l.json): rule M3`
  - `R5-75-M-H-10-s42: first rate: 410 env steps/s (8 envs, 2 workers, 4 vCPUs)`
  - `R5-75-M-H-10-s42: check at 2048 steps: N 0 (N* 0), S 15.019 s (S* 15.127 s): no gain`
  - `R5-75-M-H-10-s42: check at 4096 steps: N 0 (N* 0), S 14.688 s (S* 15.127 s): no gain`
  - `R5-75-M-H-10-s42: done at 4096 steps`
- **What `run.json` holds:**
  - status done, 4096 of 4096 steps, halt and stopped null.
  - `gates.g8a`: passes, 2 of 2, form "bar".
  - The G8a record: `at` 2026-11-17T19:38:06Z, `form_at` 2026-11-17T19:38:06Z, compared_steps 1000/1000 in both episodes.
  - `baseline_expected`: equal true, 6 episodes, digests_compared true, differ [].
  - 8 updates, each with 10 of 10 minibatch steps; lr 1e-05 → 3e-06; filter 10 Hz.
  - Two checks, each one attempt with exit 0 and a record. G8b at 2048: walk/nominal up 1 of 1, pass. No fix.
  - Session commit bb1d1f99, dirty false.

**Rule X1 gap, for you to decide (not fixed)**
On a look set the trainer never checks G9-ref itself. I probed this on a copy of the synthetic set with no G9-ref record:
- With the gates, G8a ran its 2 closed-loop episodes (record written, passes) and the 8 envs were built. Only then did the baseline's monitor refuse: `Halt: … the R0L baseline could not be made (the monitor child ended with code 8; … LookSetRefused: V6/75-M-H-10: there is no G9-ref record …)`, exit 5.
- With `--no-gates` it trained: `done at 512 steps`, exit 0, 1 update.

`g8a.py` leaves X1 to its caller, and `gates2`'s own closed-loop commands refuse a look set without a passed G9-ref. Ruling 31 (6, 7) puts the refusal in the monitor and leaves the rest to the chain. A trainer-side refusal would be a new start behaviour, so I left it to you. It does not touch S2-4, because V6/75-M-H-10 has a passed G9-ref.

**Tests**
- `tests/test_stage2_train2.py tests/test_stage2_monitor2.py` on HEAD bb1d1f99: `88 passed, 3 skipped in 311.03s (0:05:11)`.
- The three skips are the slow tests (`STAGE2_SLOW`), which I did not run: `test_tiny_r5_run_resumes`, `test_tiny_r6_run_resumes`, `test_tiny_probes_on_the_s1_view`.
- Pytest temp dirs under `s24b/` are deleted.

**Hashes**
- Bit-identity, at the end: `45719d030359ec7d5d3914af3814d837fdc05d77652ba45c64792b511027528d c0027992de21067525cc4489b74acb157feb89f3a4c7618580addc2b341be1c1`
- `train2.py` b7491e12dd87c7ffa75f03fe29908c6e11e7e33c5440ab58b4bd037109959bb6
- `monitor2.py` f7f1b180533df8008dd6c82647bab37beeb454f00b2c02dd9a2523b1a1cd8efa (unchanged)
- `clip2.py` 8e4617a8d84f9a776f218ea6f9c21036375ff6a06c2a926790233e57e82f6068 (unchanged)
- `test_stage2_train2.py` 5f049d5d5354895bfd8915f0663327d16a5c2d4c907afca91ed9d88858caa9a9
- `test_stage2_monitor2.py` 7abf7bf9245b01e9ee3b7038c8cb04d5eacffc8197eadd7a6df0c028e70694e3
- Scratch run's `run.json` 5f909be1…60ae and `g8a.json` f48c1db2…d250.

**`git diff claude/mesozoic-trex-stage2-refine-4jffre --stat`**
```
 .../trex_form_study/designlab/stage2/train2.py     | 16 +++++-
 .../trex_form_study/tests/test_stage2_train2.py    | 67 ++++++++++++++++++++++
 2 files changed, 81 insertions(+), 2 deletions(-)
```

**Commits:** 8f6adb6468bccfbddeb1f7c1fc7dd0bf39a0eec7, then bb1d1f99ed3e17d0a3c95eced94678f8b39d99d0. Both messages end with the two required lines. Branch `wt/wpc-ct` head: bb1d1f99ed3e17d0a3c95eced94678f8b39d99d0.

Files are in /tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/s24b:
- e2e/s24b_e2e.py — the driver. Its `plain` mode was added after the main run, for the X1 probe; current sha256 e83af6bf…6c64.
- logs/e2e_train.log — the run's log, sha256 4b916886…b4.
- e2e_root/out/jobs/R5-75-M-H-10-s42/ — the scratch run's job directory.
- logs/e2e_baseline.log, logs/x1_gates1.log, logs/x1_gates0.log
- x1_root/ — the X1 probe's set copy and job directories.
