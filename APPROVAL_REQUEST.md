# Tail_3 upper-bound screen: approval request

Status on 2026-10-24: **not approved, and nothing has trained.** No optimizer step has been taken on any policy. `SCREEN/APPROVED` does not exist.

SCREEN = `/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad/tailround/screen`

**The text to approve.** `SCREEN/criteria/criteria_tailscreen_final.txt`:

- sha256 `efd8e2970991ded79276d8cb37f66d31780b121d0a4bd6eed29eead8941df820`, also recorded in `criteria_tailscreen_final.sha256`;
- every pin is filled;
- APPENDIX F lists what the two reviews changed.

## What runs

| Item | Value |
|---|---|
| Arms | **CONTROL**: coef 0, with b = 0.90 for logging only. **BOUND90**: b = 0.90. **BOUND60**: b = 0.60. All three penalize the tail_3 (index 14) upper side only: coef × mean over the rows of relu(mu14 − b)², added in `MirrorKLPPO._mirror_loss` |
| Start states | stance5 **s42** at 4,194,304 (persistence: on the rail, 10/20 flagged); stance5r **s80** at 4,194,304 (onset: clean, CMD 0.6285) |
| Replicates | Seeds 101, 202, 303, the same across arms and states. 3 arms × 2 states × 3 seeds = **18 runs** |
| Horizon | One continuous run of 1,048,576 steps per replicate (4.19M → 5.24M). Checkpoints every 131,072 steps; all 9 are measured. Read at **R1 = 5,111,808** and **R2 = final** |
| Coef (`calib/coef.json`, sha256 `f6d90f99…`) | COEF_90 = **2.4**, COEF_60 = **2.4**. Pooled G_pg RMS 0.5267 over 512 mini-batches; 0.5267/0.22 = 2.394 rounds to 2.4 [measured]. Start pull on bias[14] at 2.4 [measured]: s42 BOUND90 0.083, BOUND60 1.154; s80 BOUND90 0.0015, BOUND60 0.283. Mean outward policy-gradient pull \|G_pg\| is 0.116 at s42 and 0.105 at s80 |
| Code | Scratch commits cf80ecb1..`4c0c8da7`. They change only `mirror_kl_ppo.py` and its test file. They are never pushed; the patch file has sha256 `ef0fe36b…` |

## Two choices for you (the text runs (i); either other choice means a revised text and re-pinned tools, but no new calibration)

**1. COEF level.**

- **(i)** The rule as written gives **2.4**. That is 4.8 per unit of excess, about 12% softer per row than P1's 5.47.
- **(ii)** P1's per-row stiffness gives **2.7**. Start pulls would be: s42 BOUND90 0.094, BOUND60 1.298; s80 BOUND90 0.0017, BOUND60 0.319 [inferred].

The rule's value depends on the calibration draw, by about ±10% [inferred]:

- P1's seed-42 draw gave RMS 0.602, which leads to 2.7;
- the theoretical RMS of 0.547 leads to 2.5;
- the per-rollout RMS in this calibration ranged from 0.46 to 0.60.

**2. What "per bound" means.**

- **(i)** As written: the reference excess is 0.11 above b, so COEF_90 = COEF_60.
- **(iii)** Alternative: the excess is measured from P1's absolute mean of 1.11, so it is 1.11 − b. This gives **COEF_90 = 1.3** and **COEF_60 = 0.52**. Start pulls would be: s42 0.045 / 0.250; s80 0.0008 / 0.061 [inferred]. BOUND60's dose would then be about 4.6× smaller than in (i).

## How it is read

These rules are fixed in advance. Each bound is read against CONTROL only, never against the other bound.

- **Per replicate**, from the mean of R1 and R2 (40 episodes, 20 deterministic seeds × 2 checkpoints):
  - **RAIL**: at least one tail-up flag.
  - **PARKED**: no tail-up flag, and the window-mean tail_3 command is at least 0.60. This is the literal rule; a PARKED replicate counts as NOT ADOPTED.
  - **CURED**: neither of the above.
- **Per state and per bound**, the first line that applies:
  - **s42**: RAIL (at least 1 bound replicate RAIL), PARKED (at least 2 PARKED), OTHER, NOT SEPARATED (at least 2 CONTROL replicates CURED), PASS.
  - **s80**: RAIL, PARKED, OTHER, PASS. An onset note goes beside it.
  - **OTHER** means the bound's harm episodes exceed CONTROL's. A harm episode is one where another actuator, or tail_3's lower side, saturates, or where the episode ends short. Tail-up episodes now count too.
- **Bound verdict.** If either state reads RAIL, PARKED, OTHER, or s42 reads NOT SEPARATED, the bound is **NOT ADOPTED** with that reason. Otherwise it is **PASSED**.
- **Screen outcome.**
  - **A. Both bounds PASSED**: both become candidates for a from-scratch GPU round.
  - **B. One PASSED**: that bound becomes the candidate, and the other is routed by its reason.
  - **C. Neither PASSED**: routing by reason. RAIL goes to your choice of next candidate. PARKED goes to the posture decision. OTHER goes to harm review. NOT SEPARATED means more s42 replicates, or stop.
- **When it can be read.** The scorer reads a decision only once all 18 replicates have R1 and R2, or are LOST. The decision is PROVISIONAL if any pin, gate test, panel provenance or onset label is missing. Nothing trains because of the outcome without your approval.

## Compute [inferred from measured throughput]

| Item | Estimate |
|---|---|
| Training | 6 waves of 3 runs at once; about 49–51 min per wave; **4.9–5.1 h** in total |
| Panels | 162, about 1 min each, on the 4th core |
| Gate | 10–20 min beforehand |
| Wall time | About **5.5 h** (range 5–8 h) |
| Disk | Peak about 3.7 GB of 25 GB free; the run pauses below 5 GB free |

## What happens when you approve

1. **The coordinator records the approval.**
   - It copies the final text byte for byte to `SCREEN/criteria.txt` (chmod 0444).
   - It writes `SCREEN/APPROVED`:

     ```json
     {"criteria_sha256": "efd8e297…", "runlog_line": "<your RUNLOG line>", "approved_utc": "…",
      "scratch_commit": "4c0c8da76d14f4b482f1fc63e0cf57af79b0184f", "COEF_90": 2.4, "COEF_60": 2.4,
      "coef_json_sha256": "f6d90f99fc09c09577af9dc2bf4e00fde6537cc8c4fc631af055eaf26880c18d",
      "durable_branch": "claude/handoff-tailscreen",
      "screen_common_sha256": "a478072cc385b5d49c3db2e3c0e20547d6788984558b5d3b2981c35c23ad08ea"}
     ```

   - It records the sha256 values in RUNLOG.
2. **`tools/gate.sh` runs.** It re-checks every pin and runs the whole clone test file with the gate tier enabled. It writes `gate_pass.json` only if T1b, T2, T7b and all build-tier tests are each recorded as *passed*.
3. **`tools/run_waves.sh` starts the 18 runs, 3 at a time.** Waves run in this order: (s42,101), (s80,101), (s42,202), (s80,202), (s42,303), (s80,303). Every run passes its own preflight first.
4. **The durable copy is pushed.** Once the first `run_meta.json` exists, `copy_durable.sh init` pushes the **orphan, never-merged branch `claude/handoff-tailscreen`** to origin. It is pushed from a separate repository and never opened as a PR. It carries:
   - the criteria and APPROVED;
   - `coef.json`;
   - the patch file;
   - the tools;
   - the start checkpoint pairs and stage configs;
   - RUNLOG.

   Each finished replicate then adds its data: `run_meta.json`, progress, telemetry, the R1/R2 checkpoint pairs, its panels and `panels.csv`.
5. **Panels and scoring.** Panels follow each run on the 4th core. `score_tailscreen.py` produces the decision.

## Still deferred

- **V3** trains, so it is not run before approval. After approval it may run once, as a report only: CONTROL, seed 42, from s42, 131,072 steps, compared with P1's control row. It never blocks the screen and is never a replicate.
- **The gate-tier tests (T1b, T2, T7b)** take optimizer steps on a toy model. They have been written but run only in `gate.sh`.

## What the two reviews changed

All of these are in APPENDIX F; they were checked [measured] and nothing trained.

- **T1a and T1b** now test coef 0 with rows above b. A mutation that made CONTROL train with the penalty now fails T1a.
- **OTHER** no longer hides harm on tail-up episodes.
- **Pinning.** `screen_common.py` is pinned. `gate.sh` and `run_waves.sh` are built. A skipped gate test is no longer counted as a pass.
- **Crash handling.** A crashed attempt's panels and `panels.csv` rows move away with it, and the scorer checks each row's checkpoint sha256.
- **Onset labels.** The scorer counts them.
- **Bytecode.** All runs use `-I -B`, and clone checks count ignored files.
- **Durable copy.** It now also holds the start pairs, stage configs and RUNLOG.
- **Standard error.** The SE of the R1/R2 mean is now stated as paired.

Results: the build-tier tests pass (191 passed, 3 gate-only skipped), and V1 was re-run and matches exactly. Pins and evidence are in `SCREEN/PINS.txt`.
