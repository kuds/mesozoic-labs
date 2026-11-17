Verdict: sound; body rate: 1e-5 (unchanged; rates_by_body 75-M-H-10 now rests on both heads, 75-M-H-10 and 75-M-H-10-X, each 1e-5)

## Verdict: sound. The -X head passes at 1e-5, and 75-M-H-10 stays at 1e-5 for later runs

I found no defects. Every claim I could check from the record, the code and the git state holds. I changed no tracked file and committed nothing. My check script is `/tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/xprobe/check/check_record.py`; it reads copies of the record at 9f644c0b and at HEAD saved in the same folder.

### The command
- **What ran:** `train2 probe-lr --bodies 75-M-H-10-X --set V6 --out stage2/tb/validation/lr_probe.json --workers 2`. The flags are the chain's V6 line (`run_trackb.sh:208`), except that `--bodies` names only the -X head.
- **Why that is the right form:** `probe_lr` in `designlab/stage2/train2.py` (around lines 1885 to 1990) refuses any body the record already holds ("already probed ... run once per body and head"). Passing every body as the chain does would therefore be refused for 75-M-H-10. The 60 cm sets are parked.
- **How the cell is added:** the tool merges the earlier bodies with this call's (`rows = dict(have, **out_bodies)`) and rebuilds `rates_by_body` from `base_body`. That is exactly "the way the tool does".
- **Python path:** the absolute venv path, as in the earlier V6 listing entry.

### The log and the listing
- **Log:** `stage2/tb/work/logs/V6-lr-probe-X.log` (git-ignored) shows the start at 21:34:53Z, `exit 0 wall 755.858189581 s` at 21:47:29Z, and the tool's line `{'75-M-H-10': {'rate': 1e-05, 'heads': {'75-M-H-10': 1e-05, '75-M-H-10-X': 1e-05}}}`.
- **Runner:** the runner script runs the command through `bash tools/lowprio.sh` with the required environment.
- **Listing:** the `dev_episodes.json` diff adds one V6-lr-probe entry for 75-M-H-10-X: count 294,912 (3 rates × 2 sigmas × 6 updates × 8,192 steps), listed at 21:34:53Z on commit 9f644c0b, done at 21:47:29Z. The earlier entries are unchanged.
- **Listed before the run:** the record's provenance shows the run on 9f644c0b with only `dev_episodes.json` dirty.

### The record (re-derived)
- **Hashes:** the file's sha256 is `007333feedaa51407496620334a339e9f98822ffeec8b23f25648c83fac4d9dc`. Its `sha256` field is `f433e45e170ab66b65a6b5c836de6b79e84444b8c9060e6a8d7440dff0b98272`, and that matches the record's own hash when I recompute it. The old record's field also verifies (`cccd9a92...`, file `8ee1b712...`).
- **The plain head's cell:** 75-M-H-10's cell is byte-for-byte equal to the old one.
- **Shared settings:** set, rates, sigmas, updates, seed, n_envs, n_steps, mode, parent and workers are the same in both records.
- **Top-level fields that changed:** only `added` (now [75-M-H-10-X]), `at`, `bodies`, `provenance`, `rates_by_body` and `sha256`.
- **The -X cell's setup:**
  - R5, config {}, workers 2.
  - Floors [hard ×3, carpet, hard ×3, carpet], filter 10 Hz.
  - Set freeze `77516680...47ed`. This equals freeze 0 in `stage2/sets/V6/75-M-H-10-X/set.json` and the freeze in that head's `g9_ref.json`, which has `pass_` True.
  - Its validation records the filter file's sha256 `25a61f97...`, which matches `filter.json` as it is.

Minibatch steps run per update, of 160 (`minibatch_steps_possible` is 160 everywhere):

| Head | Rate | Sigma 0.1 | Sigma 0.03 | Result |
|---|---|---|---|---|
| -X | 1e-4 | [21, 44, 52, 43, 58, 48] | [2, 3, 4, 5, 6, 6] | fails |
| -X | 3e-5 | all 160 | [14, 44, 44, 44, 59, 46] | fails |
| -X | 1e-5 | all 160 | all 160 | passes |
| plain | 1e-4 | [23, 54, 70, 73, 58, 83] | [2, 4, 5, 7, 7, 6] | fails |
| plain | 3e-5 | all 160 | [35, 78, 83, 68, 81, 50] | fails |
| plain | 1e-5 | all 160 | all 160 | passes |

- **Rule O3:** the rates were tried from the largest down and the probe stopped at the first pass, so the chosen rate for the -X head is 1e-5. My derivation matches every `passes` flag and `passed` in the record.

### The body's rate by rule O
- **The rule:** "if its rate is lower, later runs of the body use the lower one"; equivalently, the smallest passing rate over the body's probed heads.
- **The result:** min(1e-5, 1e-5) = 1e-5, so the rate does not change. `rates_by_body["75-M-H-10"]` is rate 1e-5 with heads {75-M-H-10: 1e-5, 75-M-H-10-X: 1e-5}, and `value` is 1e-5.
- **What a later run reads:** I called `train2._validation('75-M-H-10', 'V6')` read-only. It now returns rate 1e-5, heads = both, sha256 `007333fe...`.

### The commit and merge readiness
- **Commit f9dc89cc** has parent 9f644c0b and touches only the two JSON records. No code changed between 9f644c0b and HEAD.
- **Attribution:** the message ends with exactly the two required lines and names no model elsewhere (the author is "Claude <noreply@anthropic.com>").
- **Worktree:** clean; its tracked records match HEAD.
- **Main checkout:** clean, on the working branch at ca39d476.
- **Merge:** the earlier fast-forward to 9f644c0b is confirmed by the reflog. Since then the working branch has gained the lead's commit ca39d476, which touches only `checkpoint/STATUS.md`. So `wt/wp-p42` no longer fast-forwards onto it, but `git merge-tree --write-tree` gives a clean merge (exit 0, no conflicts).
- **Bit-identity:** the check prints the required two hashes now. I did not see the before and after runs that the report describes.

### Notes and what I did not verify
- **Historical references to the old hash:** `stage2/tb/runs/s2-4/.../run.json` and `roundC__do_validate75b_17nov.md` still name the old file hash `8ee1b712...`. These are records of the past and need no change, since the rate is unchanged.
- **Bundle:** the s2-b1 MANIFEST does not pin `lr_probe.json` or a rate.
- **Not verified:**
  - the load average at launch;
  - that the probe actually ran at low priority (I saw only the runner script);
  - the bit-identity output before and after the run.

I re-ran no training.
