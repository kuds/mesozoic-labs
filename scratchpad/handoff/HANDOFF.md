# Handoff: T. rex stance5 replication round (2026-10-22 to 2026-10-24)

This branch, `claude/handoff-stance5r`, is a never-merged storage branch with no repository history. It carries
the previous session's working files so the next session can continue. It holds:

- `scratchpad/`: the files, laid out as in that session's scratchpad (see below for why the path matters).
- `FILELIST.txt` and `SHA256SUMS`: every file, with its sha256.
- `restore.sh`: puts the files back and rebuilds the judge clones.
- `patches/run-training-report/`: two commits of new report code, built on main `1b71f00b` (#634's merge).
  They were never pushed as a branch.
- `NEXT_SESSION_PROMPT.md`: the prompt the maintainer pastes to start the next session.

Below, "measured" means read from files or rolled; "inferred" means reasoned from those. The full record of every
event and decision, with real UTC times, is the run log: `scratchpad/foot/stance4/live/run_log.txt` (499 lines;
lines 1-395 are already in the repo through #633 and #634).

## 1. Restore

1. Run `bash restore.sh`. It:
   - checks `SHA256SUMS`;
   - copies `scratchpad/` to `/tmp/claude-0/-home-user-mesozoic-labs/0fdf74a4-2989-56c9-a773-92101004d7de/scratchpad`;
   - re-verifies the copied files;
   - clones the four judge environments into `<scratchpad>/fix/` at their pinned commits (table below).

   The path must be exactly that one. The judges, scorers and tools hardcode it, and the criteria pin their sha256,
   so editing a path would break the pins.
2. Checkpoints (`*.zip`, `*.pkl`) are not on this branch; they stay on Google Drive (read-only). Re-fetch them only
   to re-judge, with `scratchpad/foot/stance5/replication/tools/stance5r-fetch-*.js`, whose Drive folder ids are
   filled in. Re-judging any stance5 run's common-noise test also needs the symmetry checkpoint
   `foot/symmetry/screen/runs/sym_lean_settled_s60/models/robust_best_model.zip` (sha256 c99f555c...). That file is
   from the Drive run folder `sym_lean_settled_s60`; it is not on this branch.

| Clone (`<scratchpad>/fix/`) | Branch on origin | Commit | Used by |
|---|---|---|---|
| `scratch_copband` | `claude/screen-trex-r9-armlts-copband` | `cf80ecb1` | stance5 and replication judges (arm LTSB) |
| `screen_s4_LTS` | `claude/screen-trex-r9-armlt-stagger` | `c1e146a8` | stance4 control judge |
| `screen_s3_L` | `claude/screen-trex-r9-arml-12m` | `7e6b557d` | stance3 judge |
| `screen_s3_LT` | `claude/screen-trex-r9-armlt` | `1af2d4d7` | stance3 judge |

## 2. What this round found

### The replication: decision line 2, not adopted (FINAL)

The score of record is `scratchpad/foot/stance5/replication/scores/score_final.txt` (sha256 9be2a231...) and its
`.json` (938da843...). It was produced by `score_stance5r.py --runs-ended` on 2026-10-24 at about 13:05 UTC and
prints no provisional, refusal or incomplete note.

- **s70 HOLDS.** Clean episodes at the six judged checkpoints: 40, 40, 40, 39, 40 and 40 of 40. K2c range is
  13.7 mm. K3: joint-noise falls 0 at 6.3M and 1 at final; own-noise falls 0 and 0. Roll rail 0 and 0. None of its
  91 evaluations was flagged for saturation.
- **s80 fails K1 only.** Clean episodes: 40, 4, 14, 26, 39 and 38 of 40. The 5.2M, 6.3M and 8.4M checkpoints fall
  below 33/40, all on the actuator-saturation rail (36, 25 and 14 episodes). K2, K3 (joint 1 and 2, own 0 and 0)
  and K4 (roll 1 and 1) pass.
  - The saturation alarm fired at 4,456,448. The first flagged evaluation was 4,325,376.
  - The actuator is `tail_3_pitch_act`, held at +1 (tail raised): 74% of window steps at 4.7M, 92% at 5.2M and
    63% at 6.3M.
  - s80 recovered by 10.5M.
- **H = 2 of 5** band runs hold (stance5 s52 and s70).
  - Report only, band against control: K1 3/5 vs 0/3, K2 5/5 vs 0/3, K3 3/5 vs 2/3, K4 5/5 vs 2/3.
  - Parent comparison: a TIE between stance5 s52's final and stance3 Run 2's final (40/40, 0 noise falls). s70's
    final is next (40/40, 1 noise fall).
  - Walking pilot 2's parent stays stance3 Run 2 (the maintainer's decision of 2026-10-22).
- **Not judged** (report only, not fetched to save usage after the account hit its spend limit): the 1M-grid slots
  7.3M, 9.4M and 11.5M, and the best and robust_best pairs.
- **Six-heading certificates** (report only; not gate evidence, since the band is not on main): s70's final 40/40
  seeds clean at every heading (PASS); s80's final 38/40 (PASS).
- **The menu for the next round** is stance5 criteria line 4's three items: the tail saturation habit, the dose,
  and the leg exploration-noise floor. **The maintainer has not chosen.**

### K3 mechanism readouts (report only; added on the maintainer's OK, post hoc)

The addendum `replication/readouts_addendum_k3.txt` (7a79e9f6...) fixed these readouts before any J checkpoint
existed. The tool is `replication/k3readouts/k3readout5r.py` (d0aed5a1...) with runner `k3readout_one5r.sh`
(5e3c66cd...). It was validated with 223 checked rows and 0 mismatches against stance5's skeptic-checked values,
and two adversarial reviews found nothing blocking. Outputs are in `k3readouts/out/`.

| | s70 6.3M | s70 final | s80 6.3M | s80 final |
|---|---|---|---|---|
| full noise falls of 20 | 0 | 1 | 1 | 2 |
| hip-pitch noise only / without hip pitch | 0 / 0 | 0 / 0 | 1 / 0 | 1 / 0 |
| differential / common part of the hip-pitch noise | 0 / 0 | 0 / 0 | 0 / 0 | 1 / 0 |
| +/-0.3 differential hip kick, falls of 20 | 18 | 20 | 20 | 20 |
| direct response to a L-R hip-pitch difference (/rad) | -1.02 | -0.43 | -0.48 | -0.01 |
| own hip-pitch sigma | 0.0220 | 0.0146 | 0.0212 | 0.0149 |

Reference: stance5 s42 fell 13/20 at final, with hip pitch only 15 and the differential part 15. Every stance5 band
final fell 20/20 to the kick; the stance4 control finals fell 0, 1 and 11 of 20.

Reading (inferred, small n):
- No replication run showed s42's K3 collapse. That supports a seed-42 effect for the noise falls.
- All five band finals fall to the sharp hip kick, against 0-11/20 for the control finals. That is a consistent
  band-arm property, but report only: the kick is not a criterion.
- The direct response does not predict K3 falls. s80's final is near 0 yet falls only 2/20, which agrees with the
  K3 skeptic's finding that it is a correlate, not the mechanism.

### Earlier in the round (reports, with skeptic checks first)

- `scratchpad/foot/stance5/k3diag/report.md` (8f053a92...): the K3 diagnosis. s42's noise falls are not the tail
  habit. Hip-pitch noise, mostly its left-right differential part, is the main cause. The mechanism of the
  6.3M-to-7.3M break is still open.
- `scratchpad/foot/stance5/tailexp/report.md` (de1710e6...): the scratch tail experiment, P1, a quadratic penalty on
  the policy mean beyond +/-1 (lambda 41, CPU, one seed, 1M steps).
  - Against its paired control it lowered tail saturation (mean frac99 0.26 vs 0.86) but relapsed by the end.
  - It did not keep the deterministic mean under +1.
  - Its effect against training alone is not established.
  - The code was clone `scratch_tailfix`, branch `scratch/tail-mean-bound` at `1d2d22b4`, push URL disabled:
    `environments/shared/mirror_kl_ppo.py` (+106 lines, `MeanBoundConfig`, `mean_bound_penalty`,
    `enable_mean_bound`) and a test file (+245 lines).
  - It was NEVER pushed and is not on this branch; the report describes it exactly. A real round needs the
    maintainer's approval (a training-loss change), a config key and criteria fixed in advance.
- The tail habit (`tail_3_pitch_act` at +1) has now appeared in stance4 s42, stance5 s42 and stance5r s80.

### The training-history report (code for a PR; nothing pushed)

- **Code:** `patches/run-training-report/0001`, `0002` (commits `4a783bae` and `50fd8f60` on main `1b71f00b`).
  - New module `environments/shared/reporting/training_history.py`, script
    `environments/shared/scripts/plot_training_history.py` and tests
    `environments/shared/tests/test_training_history.py` (38 tests).
  - A docs section and a CHANGELOG entry.
  - Two registry lines in `test_policy_loading.py` and `test_repository_root.py`.
- **Checks:** 158 tests pass in the touched files, and ruff 0.16.9 check and format are clean on
  `environments/`. Three reviews (numbers, code, visuals) found 3 blocking issues, all fixed.
- **What it draws:** 10 PNG pages plus HTML and JSON per run or comparison, from the run ledgers alone. It is
  report only.
- **Regenerate:** `scratchpad/foot/run_reports/regen_live.sh`. Apply the patches first; the script imports from
  `scratchpad/fix/run_report`.
- **Not done:** hip roll (not in `eval_episodes.csv`; a telemetry change), a judge-K3 overlay, and the per-step
  deep dive for finals (level 2). This is the first piece of task #47's design,
  `scratchpad/foot/stage_report/design.md`, with the maintainer's decisions of 2026-10-18.
- **Open maintainer decision: the PR's scope.**
  - (a) The standalone report as built. This is the session's recommendation as a first PR.
  - (b) Also generate it automatically at the end of every stage, as the design specifies. Better as a follow-up.

## 3. Open items and decisions waiting on the maintainer

1. **Next round:** choose from the menu (the tail habit, the dose, the leg noise floor). The tail habit is the
   evidence's lead candidate (inferred).
2. **Report PR scope:** (a) or (b) above; then open the PR from the patches.
3. **Raised by the maintainer for this session** (`scratchpad/handoff/items_for_next_session.md`): whether to add an
   actuator saturation report for final models and/or during training. Today only the scratch satwatch/satroll
   tools catch it, and the ledger does not name the actuator.
4. **Records PR**, documentation only, as #633 and #634 did. It should carry:
   - #634's landing: merged 2026-10-22 19:41 UTC; Python CI run 39831784256 on head `a7829d29`; SB3 job 18:41;
     coverage 90 percent; test (3.12, species) re-run once after an apt hang;
   - run-log lines 396 onward;
   - the replication package and its outcome (line 2), D-D41's outcome;
   - the K3 diagnosis and tail-experiment reports;
   - the K3 readouts addendum and its results;
   - the s80 saturation alarm;
   - the certificates.
5. **Walking pilot 2:** not launched. Branch `claude/pilot-walk-r9-b` stays at `8c5478a6`, held, on stance3 Run 2's
   final.
6. **Backlog:**
   - task #7, the recovery comparison;
   - task #41, blog post 2 (after the pilot verdict);
   - task #47's remaining parts.

## 4. Operational lessons from this round

- **The container rebooted** about four times and **reclaims idle sessions**. Run the queue workers as
  harness-tracked background commands, by absolute path, under `timeout 7000`, and relaunch them when they exit.
  Exit 124 is that limit, not a failure.
- **A user message that interrupts a turn can stop in-flight workflows without a notice.** Check workflow journals
  after an interrupt and resume by run id; finished agents replay from cache.
- **Usage:** the account hit its spend limit mid-run. A two-run checkpoint fetch costs roughly 0.6-0.9M subagent
  tokens. Batch fetches through `stance5r-fetch-named.js`: one call can carry several checkpoint pairs.
- **Disk:** raw Drive download dumps in the session's `tool-results` folder grew past 1 GB. Delete the ones older
  than an hour once their files are placed and hash-checked.
- **Fetch timing:** a just-written Drive file may not appear in title searches yet. The fetch agents fall back to
  folder listings.

## 5. Pins (sha256, first 16 hex)

| File | sha256 |
|---|---|
| replication `criteria.txt` | a38536de152e7359 |
| `judge/judge_stance5r.py` | 16d39f7dcb14a8b6 |
| `judge/score_stance5r.py` | 55c37ba7f30a3812 |
| `sigma_s70_s80.json` | 89778b3f81f83f05 |
| `readouts_addendum_k3.txt` | 7a79e9f66091e8fc |
| `k3readouts/k3readout5r.py` / runner | d0aed5a17bb0ea82 / 5e3c66cddaabcfab |
| `scores/score_final.txt` / `.json` | 9be2a2313747b341 / 938da8434edbbf97 |
