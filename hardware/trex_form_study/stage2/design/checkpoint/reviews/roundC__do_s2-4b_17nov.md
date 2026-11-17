S2-4 ran afresh to its budget and ended "done": no halt, no stop, no refusal, no interruption. By rule M3, S2-10's verdict is **no gain, so no go** (`go: false`). The policy learned to stay up: falls went 12, 9, 6, 1 across the checks and S rose to R0L's level. But N stayed 0 in every reading at every check, so it did not learn to walk the look. By M3, a no-gain result goes to the owner first, with both rows' failures per term. Simulation only. No code was changed and nothing was pushed.

## 1. Merges (worktree `/home/user/mesozoic-wt/wp-p42`, branch `wt/wp-p42`)
- `git merge --ff-only claude/mesozoic-trex-stage2-refine-4jffre` exited 128 ("Diverging branches can't be fast-forwarded"). wt/wp-p42 was at 58109a17 and the working branch at 5e1b949f.
- `git merge --no-ff -F <msg>` of the working branch exited 0 and made **ff31315e** with no conflicts. It added the ruling 43 text, STATUS.md and the two s2-4 reports.
- `git merge --no-ff -F <msg> bb1d1f99ed3e17d0a3c95eced94678f8b39d99d0` exited 0 and made **8b2864bd**. It changed only `train2.py` (+16/−2) and `tests/test_stage2_train2.py` (+67). 5e1b949f is an ancestor of bb1d1f99, as stated.
- Not asked: I ran the fix's two new tests (`-k "form_at or ppo_overrides"`, `--basetemp` under s24b, deleted afterwards). 2 passed in 12.3 s. They use a synthetic look set in a temporary directory, so they are no X1 episode.
- The bit-identity line, after the merges and again at the end, printed `45719d030359ec7d5d3914af3814d837fdc05d77652ba45c64792b511027528d c0027992de21067525cc4489b74acb157feb89f3a4c7618580addc2b341be1c1`. It matches.

## 2. The failed start kept as a record
- The three files were copied unchanged (cmp equal) to `stage2/tb/runs/s2-4/failed_start_17nov/`. Their sha256 match the first start's reports:
  - `run.json` 83e1a57ee43ab7847e30da252fd820360be9878ba281d6535c8cb7924c25bc54
  - `progress.json` c559f3c4511630d7cbb94197d9a175286e05dcba01b8fbc502f46ec0b088655e
  - `V8-s2-4.log` 9284ff978326ae0d318c20c1e0e71c3033a9065c1a12a82415482df6eb8ea14b
- Commit **bbf707b8**.
- Then `rm -rf stage2/tb/work/local/s2-4`. There was no V8-s2-4 marker to clear.
- I kept the old `work/logs/V8-s2-4.log` in place. The new run appended to it from line 28 on, as the chain's `step` does.

## 3. The run
- **Command:** `python -m designlab.stage2.train2 run --mode R5 --body 75-M-H-10 --seed 42 --set V6 --out stage2/tb/work/local/s2-4 --run-id s2-4 --steps 2000000 --monitor-every 500000 --workers 3 --expect-baseline stage2/tb/runs/s2-4/R0L_BASELINE.json`
  - It ran with no `--resume`, through `tools/lowprio.sh` in the background.
  - The wrapper `/tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/s24b/step.sh` is a copy of the chain's `step`.
- **Timing:** started 20:06:54Z, `exit 0` at 21:10:48Z (63 min 54 s). In `run.json`, session 0 runs 20:06:56Z to 21:10:44Z on commit bbf707b8 with dirty false. 58109a17 is its ancestor (checked).
- **Load:** the load average before the run was 0.29. Nothing else heavy ran beside it. I checked at 20:16, 20:26, 20:36, 20:46, 20:56, 21:06 and 21:10.
- **Log lines, as printed:**
  - "G8a's short form on set V6". `gates/g8a.json` has passes True, 2 of 2 identical, form "bar", with both `form_at` and `at`.
  - "baseline R0L@10ms: N 6, S 25.032 s (18 episodes, 48 s)"
  - "R0L's baseline equals the committed one (stage2/tb/runs/s2-4/R0L_BASELINE.json): rule M3"
  - "first rate: 613 env steps/s (8 envs, 3 workers, 4 vCPUs)"
  - "done at 2007040 steps"
- **Settings:** lr 1e-05 to 3e-06, linear (lr_probe.json). The env config has filter_hz 10.0.
- **Ending:** 2,007,040 steps (budget 2,007,040; asked 2,000,000) and 245 updates (`updates.jsonl` lines). Status done; halt null, stopped null; g8b null, fix null. There was no M4 reading, which comes at 5M.

**Every check.** The rule is read against R0L: N 6, S 25.032 s. A gain needs N ≥ 8 or S ≥ 26.032 s, or by M3b N_but_A1_look ≥ 8. Each check had one attempt, exit 0, with a record and `kl_cut` false.

| Check (steps) | N | N_reading2 | N_reading3 | N_but_A1_look | S (s) | Mean survival (s) | Clock factor | Read / fell / not run | Monitor wall |
|---|---|---|---|---|---|---|---|---|---|
| 507,904 | 0 | 0 | 0 | 0 | 17.677 | 17.677 | 1.00044 | 3 / 12 / 3 | 66 s |
| 1,007,616 | 0 | 0 | 0 | 0 | 22.785 | 22.785 | 1.00057 | 6 / 9 / 3 | 76 s |
| 1,507,328 | 0 | 0 | 0 | 0 | 24.064 | 24.100 | 0.99857 | 9 / 6 / 3 | 84 s |
| 2,007,040 (final) | 0 | 0 | 0 | 0 | 25.039 | 25.072 | 0.99877 | 14 / 1 / 3 | 90 s |

- The trainer printed "no gain" at each check. The first three checks are reported and decide nothing.
- Up by condition at the final check: walk/nominal 3/3, walk/v0.4 3/3, stop 3/3, nudge/left 3/3, nudge/forward 2/3 (6013 fell at 20.73 s). turn/q90_left is not run.

## 4. Collect
- `nb collect --src stage2/tb/work/local/s2-4 --dst stage2/tb/runs/s2-4 --kind run` exited 0 in 0.23 s.
- It printed "15 files (15 copied, 0 already there), each verified by sha256". It wrote COLLECTED.json, which records `dirty: True` with only its own new `jobs/` untracked.
- Commit **b78f2efd**.

## 5. S2-10
- `monitor2 s2-10 --job stage2/tb/runs/s2-4/jobs/R5-75-M-H-10-s42 --committed-baseline stage2/tb/runs/s2-4/R0L_BASELINE.json --out stage2/tb/runs/s2-4/S2_10.json` exited 0 in 0.22 s.
- It printed "S2-10 at 2007040 steps: N 0 (R0L 6), S 25.039 s (R0L 25.032 s): no gain".
- Commit **c5d84ae4**.

**The deciding numbers:**
- **N:** 0 against R0L's 6, where a gain needs 8.
- **S:** 25.0389 s against 25.0322 s, only 0.007 s more, where a gain needs +1.0 s.
- **M3b:** the survival route is closed (L 25.5577 s; L − S(R0L) = 0.525 s < 1.0 s), so M3b applies. N_but_A1_look is 0 against R0L's 6, where a gain needs 8.
- **Result:** `go: false`, `final_is_last_checkpoint: true`. `baseline_equal` is equal, with 18 episodes and digests compared.

**Failures per term at the final check:**
- Policy: along 6, line 6, look_toe_off 6, toe_clearance_own_swing 5, recovered 5, slip_median 4, cadence 3, stride 3, grf 3, look_bob 3, on_command_at_stop 3, touchdowns_after_command 2, survived 1.
- R0L: line 1, look_toe_off 3, recovered 3, survived 2, toe_clearance_own_swing 3.
- Every walk episode is read but fails along, line and look_toe_off. Every stop fails on_command_at_stop. The nudges that stayed up fail `recovered`.

## Records (sha256)
All paths are under `/home/user/mesozoic-wt/wp-p42/hardware/trex_form_study/stage2/tb/runs/s2-4/`:
- `S2_10.json` 32ebdb24e2f5568876b882379a7fc3a03044bf722becb23e210b8d78c65c9dde
- `COLLECTED.json` 6f57ef621e7bd9b4d7677b0c84a77ca1d59d7f09fe0ecfcf98e6e17b4163f0c0
- In `jobs/R5-75-M-H-10-s42/`:
  - `run.json` 3a965ec9d6bf0470514657f0fbab43f559ac213beeb63761ed1f43fb6111de3a
  - `updates.jsonl` a6af8f0b3bc4f415cb62672afa56f8f204fe2bf1de73b5c952bab66167ea2c90
  - `progress.json` a88f2fc8d28e870880cee2ab0ad470e8b67e52b95be73de8ab4d981c29728661
  - `gates/g8a.json` 8d888483423ac8787f3c9bbf365ecbd2974d3312951d0a1a779a99184e8ff83c
  - `gates/rate.json` a15c65b99d6786490fd1200705fc05f1839b982f8ba5c9d4796118709aa83ecb
  - `monitor/r0l.json` eefeea924e8342b3f17349a5be99b6fad84f6507c3e5befc61f596b8573de5e3
  - `monitor/monitor_507904.json` 472698b9b80afdb114cef178ca428dd5a67c8152f1c27946135ae19c62894748
  - `monitor/monitor_1007616.json` bdb553fa79671bace9944bd4e0b01477a1d0f892fe7b74b015a293f19abaaef9
  - `monitor/monitor_1507328.json` 996068cc20075418737f72c9aa8b2cb65c3ed0720e755ab81b4892507c5561e9
  - `monitor/monitor_2007040.json` b5141f2714c991376021c58568342ebb5ebcbbd4afb21ad2347e4d0b7a58f6c1
  - `checkpoints/checkpoint_2007040.json` 5031a705f6421e902b767e98ad6fe31f03d210b4643d7d18b9da14a644b7c554
- `R0L_BASELINE.json` is unchanged: f33e4eb11cbd25dcfb398b7a359f8704d87b2780f87879cb679220644bb70543
- The final checkpoint's model, vecnormalize and fingerprint stay uncommitted in the ignored `stage2/tb/work/local/s2-4/.../checkpoints/`:
  - model 43b12dd6cadbe9fc3db40fd1e261137cee28a1bd32cc49b9fba21ed7a4e56523
  - vecnormalize 082a77ecc3faf38132627fb31a91850b8ed5c66b8d6a82534b4fac4514071075
  - fingerprint 3e4fc4007dd77ef7a4956ac06efede206fcb1a6e0d3c2d0d27a2f08c3547ef98

## Commits on wt/wp-p42 (none pushed)
| Commit | What |
|---|---|
| ff31315e14d31bda5f6c50fb4bde448894206517 | Merge of the working branch |
| 8b2864bd7e82ff8a84bc05729d0732716b5e1190 | Merge of bb1d1f99 |
| bbf707b87f5a470ba29f433779f026030ad5e410 | The failed start kept as a record |
| b78f2efdf290fee083a4a155c00d7d33f81c095e | The collected run records |
| c5d84ae4609440ed97cc302f4607d24686e8c168 | S2_10.json |

Each message ends with the two required lines and names no model elsewhere. The worktree is clean.

## Not done, or worth knowing
- I made no `dev_episodes.json` entry, because I ran no look-set episode outside the run.
- I did not run the full test suite.
- I wrote nothing in the main checkout. Its HEAD has since moved to 4d5c104e, which is not mine.
- The chain's markers V8-s2-4, V8-collect and V8-s2-10 now exist under the ignored `work/done`. The step's commit messages are in `/tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/s24b/`.
