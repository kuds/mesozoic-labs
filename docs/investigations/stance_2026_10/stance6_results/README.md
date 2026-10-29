# The first T. rex stance round on physics r10: its results (2026-10-27 to 2026-10-28)

Evidence for §27 of
[TREX_STANCE_SYMMETRY_AND_FOOT_2026_10.md](../../TREX_STANCE_SYMMETRY_AND_FOOT_2026_10.md):
the session's run log from the first report-only snapshot of the running
files (19:13 UTC on 2026-10-27) to the start of this records change (19:32
UTC on 2026-10-28). It covers the landing of #647; the second report-only
snapshot; the runs' ends, noted from Drive at the maintainer's decision;
the final fetch, with the transfer method the maintainer decided when the
fetch brief's hand transcription was cut off; the measurements, the
completeness check and the scorer's reading of the round's three runs,
`stance6_r10LTSB_s60`, `_s42` and `_s80`, with its independent check; the
durable copy; the report-only readouts, review and next-step proposals; and
the maintainer's decisions that followed (D-D51's amendment and D-D52,
[BEHAVIOR_RECIPES_PLAN.md](../../../BEHAVIOR_RECIPES_PLAN.md) §6.2). The
centre-of-mass stance regression tests the maintainer asked for at 12:35
UTC on 2026-10-28 (#650) were a side task outside the round; their entries
are here because the log is.

The round's pre-registration is [`../stance6/`](../stance6/README.md) and the
run log of its set-up and launch is [`../stance6_setup/`](../stance6_setup/README.md).

The folders `score/` to `fetch/` hold the round's reading and what was read
after it. Every copy there is a byte copy of its session-scratch source,
absolute session-scratch paths included (no prefix, prefix2 or checkout
change, as in `../stance6/`; `review/NEXT_STEPS.txt` keeps the one checkout
path it names, as `../stance6/fetch_agent_brief.txt` does), so that its sha256
is the one the run log, the review or the round's durable copy records, but
where a copy carries the **redact** change (a session tool id or a Drive id
removed), the **lineends** change (the manifest's CRLF line ends made LF) or
the **gzip** change (a JSON over the repository's 500 KB limit for an added
file; `gunzip` restores it as it was before compression). Every source's
sha256 equals its entry in the durable copy's `SHA256SUMS` (the orphan branch
`claude/handoff-stance6` at `e4733d8`, never merged), which holds each source
byte for byte, with the rest of the round's text outputs.

[SOURCES.txt](../SOURCES.txt) lists each file's source, its sha256 and the
changes the copy made; [SHA256SUMS](../SHA256SUMS) pins them (every file of
the folder but this README).

- **`run_log.txt`**: lines 681 to 720 of the session's run log, the entries
  timed 19:13 UTC on 2026-10-27 to 19:32 UTC on 2026-10-28, with 14 workflow
  run ids (10 distinct) and 2 scheduled-task ids each replaced by
  `[id removed]` (a bare workflow id by `workflow [id removed]`). It
  continues `../stance6_setup/results/run_log.txt` (lines 600 to 680) with no
  line in both. The maintainer's decisions, with the maintainer's statements
  the records quote, are its entries timed 20:33 UTC on 2026-10-27 (#647's
  open items), 01:09 (the runs' ends noted from Drive, then the full review
  and next steps), 11:42 (preparation before the final fetch), 13:31 (the
  final fetch's transfer method), 15:05 (the durable copy's first commit
  pushed), 17:36 (#650 pushed and opened) and 19:30 UTC on 2026-10-28 (D-D52
  and this records change); the maintainer's requests of 23:59 UTC on
  2026-10-27 (the second report-only snapshot, D-D51's amendment) and 12:35
  UTC on 2026-10-28 (the centre-of-mass tests) start the work they name; the
  note of 18:40 UTC (to revisit #650's tests after the review) decides nothing
  for the round. The `STANCE6` lines are the round's records in the form its
  criteria fix: the three runs' ending notes from Drive (`TRIGGER-a` and
  `FOUND-GONE`, timed 06:25 UTC) and the final fetch's start notes
  (`F2-START`, timed 12:25 UTC). The entries timed 19:13 UTC on 2026-10-27
  and 00:39 UTC on 2026-10-28 summarise the two report-only snapshots (not
  the round's reading), the one timed 14:59 UTC the final fetch, the
  completeness check and the scorer's reading, and the one timed 19:27 UTC the
  report-only readouts, review and next-step proposals and the push of the
  durable copy's second commit (on the maintainer's word of 15:05 UTC, which
  they confirmed at 22:19 UTC covered it, after this copy ends). The copy
  ends with the entry timed 19:32 UTC on 2026-10-28, which starts this
  records change and the draft of D-D52's pre-registration; nothing later is
  in it.
- **`score/score.txt`** and **`score/score.json.gz`**: the reading, the
  output of the round's scorer (`score_r10round.py`, lane B, run once at
  14:43 UTC on 2026-10-28, exit 0): each run's completeness facts C1 to C7,
  its class, gate and outcome, the statistic A with TABLE E, and the reading
  line, `STATUS: FINAL`. The JSON is gzipped (3,542,297 bytes). The scorer
  prints four run-log lines it reads by no rule (lines 621, 667, 669 and 670,
  "for the coordinator to restate"); their workflow run ids and Drive folder
  ids are removed (in `score.txt` 2 and 2, one Drive id cut off where the
  scorer shortens its print; in the JSON, which prints the lines whole, 3
  and 2), so each reads as that line of
  `../stance6_setup/results/run_log.txt`.
- **`complete/complete_stance6_r10LTSB_<run>.txt`** and **`.json`**: the
  completeness check's output for each run (`complete_r10round.py`: C1 to C7
  all PASS, COMPLETE), the text quoting the same four lines, redacted the
  same way (2 workflow run ids and 2 Drive folder ids each). The "PRESENCE
  RAIL" lines in s42's text are the tool's own p1 to p3 figures: PRESENCE
  applies to a LOST run only, s42 is COMPLETE, and the reading counts
  PRESENCE 0 (the independent check's N3).
- **`check1/NOTES.txt`**: the independent check of the reading (14:45 to
  about 14:58 UTC on 2026-10-28), which re-derived every class, A, p and the
  line with its own code from the fetched copies and agreed; no blocking
  finding, seven notes (N1 to N7). Its scripts and outputs are on the
  durable branch.
- **`readouts/READOUTS.txt`** and **`readouts/READOUTS.json.gz`**: the
  pre-registered diagnostic readouts D-1, D-2 and D-11 to D-15, the entropy
  check and the two snapshots' context (written at 15:24 UTC; the JSON
  gzipped, 617,905 bytes). Report only. Its entropy wording at lines 802 to
  805 is narrowed by the review's §4.3 (the fixer's findings B1 and N-3).
- **`review/REVIEW.txt`**: the round's full review for the maintainer (898
  lines; written 17:52 to 18:10 UTC and revised by a fixer at 19:02 to 19:14
  UTC after three verifier lenses; sha256 `5c76f556…`, the value the run log
  gives): the per-run classes, the pre-registered readouts D-1 to D-16, the
  post-hoc entropy analysis the maintainer asked for, the comparison with r9
  and the two snapshots, the limits, and the records, among them the process
  departures (§7.4). Report only; the reading is the scorer's.
- **`review/NEXT_STEPS.txt`**: the next-step proposals (605 lines;
  proposals only; sha256 `9fd7fe64…`, the value the run log gives): five
  ranked options, the first the CPU continuation screen D-D52 takes up, and
  the pending items P1 to P4, of which P2 and P3 are this records change.
- **`review/FIXES.txt`**: the fixer's record, one entry per verifier finding
  (35 findings, 3 blocking, all fixed) and the fresh re-check's result, with
  1 workflow run id removed; **`review/HEADLINE.txt`**: the seven-line
  summary sent to the maintainer. The texts before revision, the verifiers'
  notes and the fixer's scripts are on the durable branch.
- **`kscore/kscore_stance6_r10LTSB_<run>.json`**: K1 to K4 of each run, its
  judge files scored by `kscore_r10ltsb.py` with `score_stance3.py`'s code
  (D-7, report only).
- **`d9_d10/<run>/<readout>/d9_summary.json`** and **`d10_summary.json`**:
  the summaries of D-9 (s42 only, the RAIL run: `tail_3_pitch_act` held at 0
  at 3,670,016 and at the final) and D-10 (each final under its own policy
  noise, seed 3042), run with the measuring copy's own
  `episode_trace_report.py`, report only; their per-step traces are not
  copied.
- **`fetch/manifest.csv`** and **`fetch/folders.json`**: the round's fetch
  record (271 rows: every listing, transfer and copy of F0, the launch
  checks' fetch, and of F2, the final one, with each file's size and sha256,
  and the two listings of the runs' ppo folder at 14:59:10 and 15:05:40 UTC on
  2026-10-27) and the record of the runs' Drive folders. Their 216 distinct
  Drive file and folder ids (462 times in the manifest, 4 of them also in
  `folders.json`) are removed, the sizes and sha256 left: those are the
  fetch's evidence. The durable branch keeps the ids. The manifest's CRLF line
  ends are LF here (**lineends**, a kind of change new with this folder, which
  SOURCES.txt defines), as the repository's `.gitattributes` (`text=auto`)
  stores them.
- **`panels/<run>/<label>/`** (122 files, about 13 MB; the last addition, in
  a commit of its own): for each run and each checkpoint traced (every
  1,048,576 steps, the final, and s42's onset checkpoint 3,670,016) the
  measurement panels' summaries, byte copies: `m1/m1_summary.json` and
  `m1/flags.csv` (M1, the 20-episode per-step traces reduced: the CMD_a
  figures the classes read, the tail and fall flags, hip roll), at the six J
  checkpoints `m2/m2_summary.json` (M2, the 40-episode panel: clean count,
  reward, posture) and `m4/judge_<label>.json` (M4, the judge's file, which
  `kscore/` scores), and at 6,291,456 and the final `m3/m3_summary.json` and
  `m3/stance_heading_report.txt` (M3, the six-heading certificate). The M1
  summaries at R1 and the final give the MEAN_a values `score/score.txt`
  prints.

Not copied, and in the session's scratch only (the runs' own folders also
on Drive): the final fetch's copies of the runs' files (238,406,989 bytes,
the checkpoints among them), the raw transfers, the per-step traces, the M2
panels' series and the measuring copy of the run branch. The panels'
per-episode files (M2's `panel.json`, M1's `episode_trace.json`, M3's
`stance_heading_report.json` and its CSV), the workflows' result files and
the independent check's scripts are on the durable branch.

Left as their sources have them, against the repository's pre-commit hooks:
the five D-9 and D-10 summaries and the 18 judge files end without a final
newline and 4 lines of `score/score.txt` end in spaces; leave them out of a
manual pre-commit run.
