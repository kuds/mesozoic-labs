Verdict: sound; go: no-go

**Verdict: sound. S2-10 gives no-go (`go: false`), and the rule gives the same.** I checked every claim in the report I could check against the records, git and my own tool output, and each one holds. I changed no tracked file, committed nothing and pushed nothing. Simulation only.

**What I ran myself.** My scripts and outputs are in `/tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/s24b/check_run/`:
- `rederive.py` re-derives the readings from the raw episode tables. It does not import designlab. Its output is in `rederive.out`.
- `cmp_s210.py` compares my re-run S2-10 record with the committed one.
- `S2_10.rerun.json` and `s2_10_rerun.log` are the re-run's output.

**Merges.**
- The reflog shows wt/wp-p42 at 58109a17 and the working branch at 5e1b949f when the merges ran (20:05:37 and 20:05:45). 58109a17 was only on wt/wp-p42, so the ff-only merge could not succeed and a real merge was right.
- ff31315e (parents 58109a17 and 5e1b949f) adds only the ruling 43 text, STATUS.md and the two s2-4 reports.
- 8b2864bd changes only `train2.py` (+16/−2) and its test (+67). That equals `git diff 5e1b949f bb1d1f99` exactly.
- wt/wpc-ct is at bb1d1f99, and the fix's check report (in 4d5c104e) gives "Verdict: merge" at that tip.
- `git diff bb1d1f99 c5d84ae4` touches only `stage2/tb/runs/s2-4/` (21 files). So the merge of bb1d1f99 is the only code change; everything else is a record.
- The bit-identity line, run in the worktree, prints `45719d03…528d c0027992…c1c1`. That matches.

**The failed start kept as a record.** The sha256 of the three files under `failed_start_17nov/` match the first start's reports:
- `run.json` 83e1a57e…5bc54
- `progress.json` c559f3c4…8655e
- `V8-s2-4.log` 9284ff97…ea14b

The failed start has run_uuid 06026565…, status partial, commit 58109a17, and the TypeError as its error. There is no V8-s2-4 marker from it: the chain writes a marker only on exit 0, and run_trackb.log shows "FAILED V8-s2-4 (exit 1)".

**The run.**
- **Command:** the V8-s2-4 log (line 28) holds the asked command word for word, including `--workers 3 --expect-baseline stage2/tb/runs/s2-4/R0L_BASELINE.json` and no `--resume`.
- **Wrapper:** `step.sh` does what the chain's `step` does (lowprio, the log, the marker on exit 0).
- **Timing:** start 20:06:54Z, "exit 0" at 21:10:48Z.
- **run.json:**
  - New run_uuid 18d46b99…
  - One session, from 20:06:56Z to 21:10:44Z, on commit bbf707b8 with dirty false. 58109a17 is an ancestor of that commit.
  - resumed_from None and restarted_from_zero False, so there was no interruption.
- **Settings match the validation records:**
  - lr runs from 1e-05 to 3e-06, linear. Its source is lr_probe.json (sha256 8ee1b712…), whose rates_by_body is 1e-05; the end is 0.3 × that rate.
  - filter_hz is 10.0 in all 8 env configs. filter.json (25a61f97…) chose 10 Hz.
  - Seed 42, head current, 8 envs, 3 workers, latency_steps [0, 12].
- **Baseline:** `baseline_expected` says equal true, 18 episodes, digests compared, at 20:08:55Z. The first rollout came at 20:09:08Z (rate.json), so the check was made before it.
- **Ending:**
  - Status done; halt, stopped and fix are null; g8b is null.
  - num_timesteps is 2,007,040 = 245 × 8192, and `updates.jsonl` has 245 lines.
  - The last update has lr 3e-06; kl_cut is false throughout.

**Every check.** Each had one attempt, exit 0, with a record that names the checkpoint's hashes. My own recomputation gives:

| Check (steps) | N | Reading 2 | Reading 3 | N_but_A1_look | S (s) | Mean survival (s) | Clock factor | Read / fell / not run |
|---|---|---|---|---|---|---|---|---|
| 507,904 | 0 | 0 | 0 | 0 | 17.676667 | 17.677 | 1.000438 | 3 / 12 / 3 |
| 1,007,616 | 0 | 0 | 0 | 0 | 22.784667 | 22.785 | 1.000568 | 6 / 9 / 3 |
| 1,507,328 | 0 | 0 | 0 | 0 | 24.064324 | 24.100 | 0.998571 | 9 / 6 / 3 |
| 2,007,040 | 0 | 0 | 0 | 0 | 25.038929 | 25.072 | 0.998769 | 14 / 1 / 3 |

- No check shows a gain against N* 6 and S* 25.032222 (R0L's, plus every earlier check).
- The monitor wall times printed in the log are 66, 76, 84 and 90 s.

**S2-10, re-derived.**
- R0L (committed): N 6, S 25.032222 s, L 25.557704 s.
- The run's own `r0l.json` equals the committed baseline in all 18 episodes (condition, seed, outcome, survival_s, state_digest). Every episode that ran holds both digests.
- **N:** 0 against 6. A gain needs 8, so no.
- **S:** 25.038929 s against 25.032222 s, which is +0.006707 s. A gain needs 26.032222 s, so no.
- **M3b:** L − S(R0L) = 0.525482 s, which is under 1.0 s, so the survival route is closed and M3b applies. N_but_A1_look is 0 against R0L's 6, and a gain needs 8, so no.
- **Result:** `go = False`.
- **Failures per term, policy:** along 6, line 6, look_toe_off 6, toe_clearance_own_swing 5, recovered 5, slip_median 4, cadence 3, stride 3, grf 3, look_bob 3, on_command_at_stop 3, touchdowns_after_command 2, survived 1.
- **Failures per term, R0L:** line 1, look_toe_off 3, recovered 3, survived 2, toe_clearance_own_swing 3.
- **Up by condition:** walk/nominal 3/3, walk/v0.4 3/3, stop/two_step 3/3, nudge/left 3/3, nudge/forward 2/3 (seed 6013 fell at 20.73 s). turn/q90_left is not run.

In plain words: the policy learned to stay up. Falls went 12, 9, 6, 1 across the checks, and S rose to R0L's level. But no episode succeeded in any reading at any check. Every walk episode fails along, line and look_toe_off, so it does not walk the commanded path, with or without the look. So by M3 it is no gain and no go, and it goes to the owner first with both rows' failures per term.

**My re-run of S2-10** (into scratch, from the worktree's study directory, with the chain's environment, through lowprio):
- It exited 0 in 0.41 s and printed "S2-10 at 2007040 steps: N 0 (R0L 6), S 25.039 s (R0L 25.032 s): no gain".
- Its record differs from the committed `S2_10.json` only in `at`, in `provenance` (it ran at commit c5d84ae4; the committed one at b78f2efd) and in the record's own content hash. Its sha256 is e0f22fdd…da5d.

**Records.** I recomputed every sha256 the report lists, and each matches:
- S2_10.json 32ebdb24…9dde; COLLECTED.json 6f57ef62…f0c0; R0L_BASELINE.json f33e4eb1…0543 (equal to the blob at 58109a17).
- In `jobs/R5-75-M-H-10-s42/`: run.json 3a965ec9…de3a, updates.jsonl a6af8f0b…ea90, progress.json a88f2fc8…8661, gates/g8a.json 8d888483…e83c, gates/rate.json a15c65b9…3ecb, monitor/r0l.json eefeea92…e5e3.
- The monitors: 472698b9…4748 (507,904), bdb553fa…aaef9 (1,007,616), 996068cc…61e9 (1,507,328), b5141f27…f6c1 (2,007,040).
- checkpoint_2007040.json 5031a705…c554.
- The final checkpoint's model 43b12dd6…6523, vecnormalize 082a77ec…1075 and fingerprint 3e4fc400…ef98 are in the ignored work directory, and the checkpoint's record names the same three hashes.

The 15 collected files are byte-equal (cmp) to their originals in `work/local`. The report does not list the four other collected checkpoint records:
- checkpoint_507904.json 5c04be6d…051f
- checkpoint_1007616.json 87efe4f7…5881
- checkpoint_1507328.json 5ed560f5…051f
- checkpoint_1753088.json 16befc36…c0c6 (a 250k checkpoint the run kept)

**Commits.** These five are on wt/wp-p42:
- ff31315e (merge of the working branch)
- 8b2864bd (merge of bb1d1f99)
- bbf707b8 (the failed start kept as a record)
- b78f2efd (the collected run records)
- c5d84ae4 (S2_10.json)

Each message ends with exactly the two required lines and names no AI model elsewhere ("model files" in b78f2efd means the policy's files). No remote branch contains them. The worktree and the main checkout are clean.

**Ready to merge.** Read-only `git merge-tree` of wt/wp-p42 into the working branch (now at 4d5c104e, merge base bb1d1f99) exited 0 with no conflicts.

**What I could not verify:**
- The load of 0.29 before the run and the doer's polling times are not in any record. rate.json shows load averages of 1.36 and 1.58 during the first rollouts, which fits the run alone on 4 vCPUs.
- The only other file written in the scratch area during the run is the lead's light 20:12 commit message.
- I could not see when the doer's two extra fix tests ran.

**Not a defect of this step:** the chain's V8 note still says the trainer takes no committed baseline, but since ruling 17 it does. The lead's command adds `--expect-baseline`, and the doer followed that command.

I did not run pytest, and I ran no closed-loop episode.
