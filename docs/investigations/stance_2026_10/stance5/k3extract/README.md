# The band runs' judge files, cut to K3 (written 2026-10-27)

A compact per-run extract of the judge files of the five T. rex stance runs
of arm LTSB on physics r9 (`stance5_LTSB_s52`, `_s60` and `_s42`, judged by
the stance5 judge, and `stance5_LTSB_s70` and `_s80`, judged by the stance5
replication judge), the maintainer's decision of 2026-10-25 17:42 UTC (D-D46,
[BEHAVIOR_RECIPES_PLAN.md](../../../../BEHAVIOR_RECIPES_PLAN.md) §6.2): the
training-history report's judge-K3 readout reads a run's judge files, which
stay in the session's scratch, so this extract puts that readout's inputs in
the repository (the report itself still reads each run's own stage
directory, its ledgers and probe, which are not here). It holds the judge's
recorded K3 inputs only (no gate verdict, panel or readout of its own) and
changes no file the replication scored.

- **`<run id>/judge_<steps>.json` and `<run id>/judge_final.json`** (39
  files, 7 to 9 per run): each judge file the readout reads (none of a run's
  `judge_best.json`, `judge_robust_best.json`, `judge_common_noise_final.json`,
  `judge_fresh_final.json`, `judge_tail_final.json` or `judge_quick_*.json`),
  cut to the keys it reads and written with indent 1: `schema`, `mode`,
  `smoke`, `noise_run`, `seed`, `steps`, `checks.checkpoint_num_timesteps`,
  and of `noise_matched_control_joint` (the joint-space noise test) and
  `noise_matched_own` (the own-sigma test) `n`, `falls`, `fall_seeds`,
  `seeds`, `sigma`, `sigma_vector` and `episodes_detail` with each episode's
  `seed`, `length` and `reached_horizon`. Every value is the judge file's own.
  `plot_training_history.py <stage dir> --judged
  docs/investigations/stance_2026_10/stance5/k3extract/<run id>` (from the
  repository root) reads a run's directory here as it reads the judge's.
- **`k3_extract.csv`**: one row per file, in run and step order: the run, its
  seed, the judge's schema, the file, its slot and step (the final's from
  `checks.checkpoint_num_timesteps`, 12,001,280), the joint-space and
  own-sigma falls of `n`, the spawns (3042-3061) and the source file (below
  the session's scratch root) with its sha256.
- **`make_k3extract.py`**: the script that wrote both from the judge files
  (`python3 make_k3extract.py <scratch root> <out>`; Python's standard
  library only).

[SOURCES.txt](../../SOURCES.txt) gives each file's source and sha256;
[SHA256SUMS](../../SHA256SUMS) pins every file but this README.

Checked on 2026-10-27 (measured): for each run the readout of this extract
equals the readout of the judge's own directory, point for point
(`read_judge_k3`, the readout's files listed as not read aside), and its K3
checkpoints (6,291,456 and the final) give, for all five runs, the replication
score of record's K3 falls and its K3 verdicts (held for s52, s70 and s80; not
for s60 and s42)
([`../replication/results/scores/score_final.json.gz`](../replication/results/scores/score_final.json.gz)).
`environments/shared/tests/test_training_history.py`
(`test_the_committed_k3_extract_reads_as_its_index_and_the_scored_k3`) checks
the extract against `k3_extract.csv` and the score of record in the suite.
The stance4 control runs, which the replication also scored, are not
extracted.
