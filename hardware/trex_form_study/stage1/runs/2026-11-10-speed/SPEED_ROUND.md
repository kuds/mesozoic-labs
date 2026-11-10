# Stage 1b speed round: how this run directory is made

The plan is `STAGE1B_PLAN.md`; its section 6 holds the decisions taken before the runs. The code is
`designlab/stage1/speedfit.py` (the fit, speed G5 and the validation rows) and `designlab/stage1/speedreport.py` (plan
section 4's reading of the rows). `run_speed_round.sh` holds the exact commands; run it from
`hardware/trex_form_study` in the main checkout.

## Before anything runs

1. Merge this branch and the faster-knee branch. The speed fit's sources include modules both branches edit, so no fit
   starts before the merge, arm S's included (decision 5).
2. Run arm K's `prepare` and gates G1–G4 (`run_arm_k.sh`). Their references and transitions (`<body>/refs/*.npz`,
   `<body>/transitions/*.npz`) and the case's models (`round2/refine/narrow_knee_60/models`) are git-ignored, so the
   merge does not bring them: copy them into the main checkout from the worktree that made them, or run
   `run_arm_k.sh` there.
3. Run `bash stage1/runs/2026-11-10-speed/run_speed_round.sh check`. It checks that the K bodies are registered and
   their gates recorded, and that every body's git-ignored inputs are here (the round-two models, and each reference
   and transition the round reads, against its recorded sha256). It prints the K bodies' push bounds and the speed
   sources' sha256, and runs `test_stage1_speedfit`. Record that sha256: none of the 14 modules may change until the
   rows have run.

## Per arm

| | Arm S: `60-N-T-10`, `60-N-H-10` | Arm K: `60-N-T-10-K`, `60-N-H-10-K` |
|---|---|---|
| References | Stage 1's run: `--inputs stage1/runs/2026-11-08` | this run (no `--inputs`) |
| Start | the body's own Stage 1 G5 selection | the pick's: `--start-from 60-N-T-10` / `60-N-H-10` |
| Rows | `R2s/42`, `R2s/42@10ms`, `R2@10ms` | the same and `R2@0ms` |
| Q1 / Q2 baseline at 0 ms | Stage 1's `R2/<start seed>` on the body | `R2@0ms` on the body |

The rows read their gains from this directory (`STAGE1B_SPEED_RUN`, set by the script; the default is the same). Every
row runs with `--attribute`, so its falls get decision 20's labels (decision 2).

## Order and budget

Run the four bodies at once, one core each, with nothing else heavy on the machine:
`run_speed_round.sh all BODY` runs the fit, then speed G5, then the rows. Each step continues after a restart. Then
run `run_speed_round.sh report`, which writes `STAGE1B_TABLES.md` and `stage1b_tables.json` here.

`60-N-T-10/speedfit/cost.json` holds the cost of one evaluation and the projections (`speedfit.COST_RULE`). If the
fits cannot reach generation 30 in time for speed G5 and the rows before 11:00 UTC, stop them and judge every arm at
the last generation all of them reached: `run_speed_round.sh g5 BODY G`, then `rows BODY` (plan section 5).

## Files

- `<body>/speedfit/start.json`: the body's start, recorded once.
- `<body>/speedfit/42/cma.json`, `evals.jsonl`: the fit's checkpoint and its evaluation log.
- `<body>/speedg5.json`, `speedg5_evals.jsonl`: speed G5 and its kept validation evaluations.
- `<body>/eval/<row>/<behavior>/<condition>.json`: the rows' panel records. The panels' `eval/summary.json` gives
  Stage 1's verdicts, which do not apply to these rows; read `STAGE1B_TABLES.md` instead (decision 3).
- `logs/`: the script's logs (not committed).
