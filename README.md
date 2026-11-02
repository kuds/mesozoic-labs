# Phase-critic Colab round (PC, "STANCE7"): results durable copy (never merged)

The finished round of the T. rex r10 LTSB stance study with a critic-only episode phase input (branch
claude/screen-trex-r10-phasecritic at 1dfe2327, pushed, never merged; pre-registration on
claude/handoff-phase-critic-prereg). Six 12M-step Colab L4 runs: three PHASE (critic phase on) and three OFF (switch
off), seeds 60, 42, 80. Text only; the fetched run files stay on Drive.

Start with final/:
- final/PC_REPORT.md: the round report (both readings, per-run table, the independent checks, limits, process notes).
- final/ALL_EXPERIMENTS_SUMMARY.md: every tail-habit experiment, oldest first, and the causal story with labels.
- final/NEXT_STEPS.md: the recommendation and decision tree after this round, with the maintainer's open decisions.

The readings:
- PINNED (the reading of record, f2/score/score_pc.txt, STATUS: FINAL): line 4, "NO TAIL RAIL OR PARK IN 3 OF 3
  PHASE-CRITIC RUNS, THE STANCE GATE CERTIFICATE PASSED IN 3 OF 3, NO SATURATING HABIT ON ANOTHER ACTUATOR: CONSISTENT
  WITH THE PHASE CRITIC REMOVING THE HABIT, NOT CONFIRMATORY". THE ONE TEST (pinned): NOT SEPARATED (p = 20/20): a
  phase run's output was read at 2026-11-01 17:44 UTC before any OFF run launched, so the OFF runs count as missing.
- FLAGGED DEVIATION (amend2/, decided 2026-11-02 13:52 UTC and approved 15:59:28 UTC, both before F2;
  f2/score/score_pc_dev.txt): THE ONE TEST as if the OFF runs were blind reads LOWER, p = 1/20.
- Measured: the three OFF runs are byte-for-byte reproductions of the stance6 runs of the same seeds (training is
  deterministic on that Colab image), so the OFF arm adds no new sample (final/PC_REPORT.md).

Contents: round/ (criteria.txt, APPROVED, F2_PINS.json, launch checks, the F0 store, the fetch manifest, panels with
the measurement summaries), f2/ (prep, per-run measure logs, score/, check_numbers/, check_process/), f2prep/ (VP1,
VP4, the F2 pins draft, disk report, deletions), amend2/ (the deviation reading: AMENDMENT2.txt, the tool, tests,
reviews, APPROVED2.json), final/, COLAB_STEPS.txt, run_log.txt (the session's run log as it stood when this copy was
made). skipped.txt lists what was left out.

Built 2026-11-02 20:35 UTC. FILELIST.txt and SHA256SUMS cover every file (sha256sum -c SHA256SUMS).
