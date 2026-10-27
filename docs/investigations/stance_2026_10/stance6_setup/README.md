# The first T. rex stance round on physics r10: its set-up (2026-10-25 to 2026-10-27)

Evidence for §26 of
[TREX_STANCE_SYMMETRY_AND_FOOT_2026_10.md](../../TREX_STANCE_SYMMETRY_AND_FOOT_2026_10.md):
the session's run log from the heading-test fix's green CI run (08:21 UTC on
2026-10-25) to the round's launch, its launch checks and the next decision.
It covers the landings of the heading-test fix and physics r10 (D-D44) and
of the two report-only telemetry changes, the
maintainer's decisions D-D45 to D-D51
([BEHAVIOR_RECIPES_PLAN.md](../../../BEHAVIOR_RECIPES_PLAN.md) §6.2: walking
pilot 2 retired; the telemetry, no MuJoCo sensor and the in-run K3 panel;
the round's design, its all-or-nothing run handling, its approval and its
launch), the version decision (`0.4.0.dev2`), the criteria drafts and their
checks, the run branch, and the launch of the round's three runs,
`stance6_r10LTSB_s60`, `_s42` and `_s80`, on 2026-10-27. The round's runs
were training when this copy was taken (2026-10-27, at 19:01 UTC): no result
of the round is in it, and none is recorded here.

[SOURCES.txt](../SOURCES.txt) lists each file's source, its sha256 and the
changes the copy made; [SHA256SUMS](../SHA256SUMS) pins them (every file of
the folder but this README).

- **`results/run_log.txt`**: lines 600 to 680 of the session's run log, the
  entries timed 08:21 UTC on 2026-10-25 to 18:41 UTC on 2026-10-27, with 47
  workflow run ids (23 distinct), 3 scheduled-task ids, 1 background-job id,
  1 id of the session's task list (a '#' and a number) and 2 Drive folder ids
  each replaced by `[id removed]` (a bare workflow id by
  `workflow [id removed]`), and the session's checkout path by `checkout`.
  Its first four lines (600 to 603) are the last four of
  [`../stance5/tail_sr3/results/run_log.txt`](../stance5/tail_sr3/results/run_log.txt)
  (lines 578 to 603), byte for byte. The maintainer's decisions, with the
  maintainer's statements the records quote, are its entries timed 14:13,
  14:30, 14:45, 15:44, 16:01, 16:03, 16:15, 17:42, 19:01, 20:13 and 22:18 UTC
  on 2026-10-25, 01:12, 12:54 (two), 13:06, 13:08 (a remark), 15:47 and 21:21
  UTC on 2026-10-26, and 01:32, 05:55 (the approval), 15:55 (the launch
  report) and 18:41 UTC on 2026-10-27; the maintainer's other requests (19:22
  UTC on 2026-10-25, to be told when two pieces of work were ready; 15:57,
  17:00 and 18:38 UTC on 2026-10-27, the Drive checks below; 18:35 UTC, this
  records change) decide nothing; the one of 12:06 UTC on 2026-10-25 is
  D-D44's amendment, recorded with D-D44. The `STANCE6` lines (timed 14:57,
  15:32 and 15:54 UTC on 2026-10-27) are the round's launch and launch-check
  records in the form its criteria fix: configuration only. The entries timed
  15:57 and 18:38 UTC are liveness checks by listing only (names, sizes and
  times of files; no content read), and the one timed 17:00 UTC confirms from
  the listing that each run writes its in-run K3 panel files. The last entry
  (18:41 UTC) starts a report-only snapshot of the running files at the
  maintainer's request (D-D51's amendment); the copy ends before anything it
  read.
- **`results/stance6_launch_lines.txt`**: the run log's six `STANCE6` lines on
  their own, its lines 666, 668 and 671 to 674, the same bytes as those lines
  of `results/run_log.txt`: the three `LAUNCH` lines (each run's settings as
  launched) and the three `CHECK` lines (each run's launch check,
  configuration only, all three no-fault), with three workflow run ids (two
  distinct) removed. They were written after the launch, from 14:57 to 15:54
  UTC on 2026-10-27, five of them after the runs' sessions had started (15:01
  to 15:03 UTC), and are kept here, with the run log.
