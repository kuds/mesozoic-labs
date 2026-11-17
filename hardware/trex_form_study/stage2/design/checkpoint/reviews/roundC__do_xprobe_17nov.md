## Rule O3's learning-rate probe on 75-M-H-10-X, V6 (ruling 44): the -X head passes at 1e-5, and the body's rate for later runs stays 1e-5

The probe exited 0 and the record is committed as **f9dc89cc170adc79e688cd0baf6fd320c666150c** on `wt/wp-p42`. Nothing was pushed.

**Command**, run from `/home/user/mesozoic-wt/wp-p42/hardware/trex_form_study` with the required environment, through `bash tools/lowprio.sh` in the background:
`/home/user/mesozoic-labs/hardware/trex_form_study/.venv/bin/python -m designlab.stage2.train2 probe-lr --bodies 75-M-H-10-X --set V6 --out stage2/tb/validation/lr_probe.json --workers 2`

- **Exit:** 0.
- **Wall time:** 755.9 s (21:34:53Z to 21:47:29Z).
- **Tool's output line:** `lr probe on set V6: {'75-M-H-10': {'rate': 1e-05, 'heads': {'75-M-H-10': 1e-05, '75-M-H-10-X': 1e-05}}}`
- **Settings in the -X cell:** fresh R5 model, seed 42, 8 envs on floors [hard ×3, carpet, hard ×3, carpet], filter 10 Hz (from `filter.json`), 6 updates per cell, 2 workers, set freeze `77516680b66203cff886be35fc4527eb47222cc22017dab89d6176f94f5747ed`.

### Minibatch steps run per update (of 160)

| Head | Rate | Sigma 0.1 | Sigma 0.03 | Result |
|---|---|---|---|---|
| 75-M-H-10-X (new) | 1e-4 | [21, 44, 52, 43, 58, 48] | [2, 3, 4, 5, 6, 6] | fails |
| 75-M-H-10-X (new) | 3e-5 | all 160 | [14, 44, 44, 44, 59, 46] | fails |
| 75-M-H-10-X (new) | 1e-5 | all 160 | all 160 | **passes** |
| 75-M-H-10 (kept as it was) | 1e-4 | [23, 54, 70, 73, 58, 83] | [2, 4, 5, 7, 7, 6] | fails |
| 75-M-H-10 (kept as it was) | 3e-5 | all 160 | [35, 78, 83, 68, 81, 50] | fails |
| 75-M-H-10 (kept as it was) | 1e-5 | all 160 | all 160 | passes |

I checked that 75-M-H-10's cell in the new record is identical to the committed one.

### The body's rate by rule O

- **Chosen rate for the -X head:** 1e-5.
- **75-M-H-10's rate for later runs:** stays 1e-5. Rule O says later runs use the -X head's rate only "if its rate is lower", and 1e-5 is not lower than the body's 1e-5. The record now gives `rates_by_body` → `75-M-H-10`: rate 1e-5, resting on both heads (`75-M-H-10` and `75-M-H-10-X`), and `value` is 1e-5.
- **Why it could not have gone lower:** 1e-5 is the smallest rate the probe tries. The only other possible outcome was no passing rate, which would have made the body's rate None and sent the body to you.

### Record and listing

- **Record:** `stage2/tb/validation/lr_probe.json`, with `added` = [75-M-H-10-X].
  - The `sha256` field inside the record is `f433e45e170ab66b65a6b5c836de6b79e84444b8c9060e6a8d7440dff0b98272`, and it matches the record's own hash.
  - The file's sha256 is `007333feedaa51407496620334a339e9f98822ffeec8b23f25648c83fac4d9dc`.
- **Listing:** `stage2/tb/dev_episodes.json` has a new entry from `gates2 dev-episode`:
  - Step V6-lr-probe, set V6, body 75-M-H-10-X, "R5 training env steps", count 294,912, reason "rule O3's learning-rate probe".
  - Listed at 21:34:53Z, before the run (commit 9f644c0b); marked done at 21:47:29Z.
  - The plain head's entry is unchanged.

### Process

- **Merge:** fast-forwarded `wt/wp-p42` from b38be399 to `claude/mesozoic-trex-stage2-refine-4jffre` at 9f644c0b.
- **Bit-identity:** the check printed the required two hashes before and after the run.
- **Machine:** idle at launch (load 0.17); load was about 1.5 during the run.
- **Commit contents:** only the two record files. No code was changed, nothing was pushed, and the main checkout was not touched.
- **Log:** `/home/user/mesozoic-wt/wp-p42/hardware/trex_form_study/stage2/tb/work/logs/V6-lr-probe-X.log` (git-ignored).
- **Runner script:** `/tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/run_xprobe.sh`.

**Not done:** I did not run the chain itself (`run_trackb.sh`), so its done-marker files in `stage2/tb/work/done` were not touched. I did not probe the 60 cm heads.
