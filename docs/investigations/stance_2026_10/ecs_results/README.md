# The entropy continuation screen and the tail-habit work after it (2026-10-28 to 2026-10-29)

Evidence for decision D-D52's outcome and decisions D-D53 to D-D55
([BEHAVIOR_RECIPES_PLAN.md](../../../BEHAVIOR_RECIPES_PLAN.md) §6.2): the
session's run log from the maintainer's decisions on the entropy continuation
screen's design (21:05 UTC on 2026-10-28) to the start of this records change
(23:21 UTC on 2026-10-29). It covers the screen's build, approval, launch,
24 CPU continuations, reading and independent checks; the landing of #652;
the durable copy of the screen's text outputs; and the work on the T. rex
tail saturation habit that followed the reading: the centre-of-mass revisit,
the tail clamp test, the normaliser probe, the posture tests and the record
of everything tried.

The screen continued two checkpoints of the first r10 stance round, whose
results are [`../stance6_results/`](../stance6_results/README.md).

[SOURCES.txt](../SOURCES.txt) lists each file's source, its sha256 and the
changes the copy made; [SHA256SUMS](../SHA256SUMS) pins them (every file of
the folder but this README).

- **`run_log.txt`**: lines 721 to 759 of the session's run log, the entries
  timed 21:05 UTC on 2026-10-28 to 23:21 UTC on 2026-10-29, with 22 workflow
  run ids (11 distinct) and 1 background-job id each replaced by
  `[id removed]` (a bare workflow id by `workflow [id removed]`). It
  continues `../stance6_results/run_log.txt` (lines 681 to 720) with no line
  in both. Copy line = source line - 720.
  - The maintainer's decisions, with the maintainer's statements the records
    quote, are its entries timed 21:05 UTC on 2026-10-28 (the screen's design:
    D1 to D9 as recommended, D10 to D12 off), 22:19 UTC (#652 pushed and
    opened), 01:43 UTC on 2026-10-29 (the screen approved and launched), 12:34
    (the screen's durable copy pushed; next, the centre-of-mass follow-up),
    12:42 and 13:10 (the tail clamp test, both ways), 17:28 (the
    centre-of-mass test change, the creep scan, CoM-P1 report-only in the next
    round, the normaliser probe), 18:18 (one record of everything tried on the
    tail habit), 19:55 (three posture tests) and 23:21 UTC (the record as its
    own note with this records change). Those of 2026-10-29 are D-D52's
    amendment and D-D53 to D-D55.
  - The screen's records: the build (21:08 UTC on 2026-10-28 and 01:42 UTC
    on 2026-10-29), the launch (01:44 UTC), the check-ins (03:41 to 10:58 UTC;
    the last of the 8 waves of 3 was decided by 12:05 UTC), the reading (12:17
    UTC), its independent check (12:33 UTC) and the durable copy (12:35 UTC).
    The entry timed 12:33 UTC, the copy's line 22, corrects two of them in
    words: line 18 (the source's 738) should read "6 continuations left: 3
    running, 3 queued", and line 20 (the source's 740) omits the reading
    line's pre-registered parenthesis, "(not tested: a shift of the first flag
    within the window, a modest effect, the early phase)". The copy keeps all
    three as the source has them.
  - The entries timed 12:38, 12:40, 16:05, 17:26, 17:30, 20:11, 20:55, 22:39
    and 23:20 UTC are the centre-of-mass revisit, the coordinator's static
    check of the tail as a counterweight, the clamp test, the follow-ups (the
    centre-of-mass test change, the creep scan and the normaliser probe), the
    record's draft, a container restart, the posture tests and the record's
    update. The entries timed 22:17 and 22:28 UTC on 2026-10-28 and 00:28 UTC
    on 2026-10-29 are #652's build, push and landing. The copy ends with the
    entry timed 23:21 UTC on 2026-10-29, which starts this records change;
    nothing later is in it.

## The screen's records

The entropy continuation screen (ECS) continued `stance6_r10LTSB_s42` from
its 3,145,728-step checkpoint and `_s80` from its 4,194,304-step checkpoint
for 1,572,864 steps each, on the session's CPUs: arm A with the entropy
coefficient as the runs recorded it, arm B with it at 0, 6 continuations per
arm and start (24). Its approved criteria fix one statistic: whether a
continuation has at least one ANY-flagged evaluation by its end, read by an
exact stratified test, two-sided at 0.025 per side. The scorer read, final:
line 3, "NO DIFFERENCE DETECTED IN WHETHER AN ANY-FLAGGED EVALUATION CAME BY
E FROM THESE STATES (not tested: a shift of the first flag within the
window, a modest effect, the early phase)": s42 arm A 6 of 6, arm B 6 of 6;
s80 arm A 1 of 6, arm B 1 of 6; P_UP = P_DOWN = 17/22. Two independent
checks agree. It certifies nothing.

Each file below is a copy of the file at the same path in the screen's scratch
folder (`scratchpad2/r10round/ecs/`), but `run_logs/` and `build_notes/` (the
scratch `logs/` and `build/`; the repository's `.gitignore` ignores every
directory named `logs` or `build`). Every copy is byte-identical to its
source, absolute session-scratch paths included, as in `../stance6/`, so that
each sha256 is the one the approval, the screen's `SHA256SUMS.txt` or its
durable copy records, but for **redact** (the one workflow run id in each of
`FOR_THE_MAINTAINER.txt`, `survey_design.txt`, `v2/decisions_ecs_v2.txt`,
`build_notes/LANE_R.txt` and `build_notes/FIXES.txt`) and **gzip**
(`tools/run_ecs.py`, below). `criteria.txt` and `final/criteria_ecs_final.txt`
keep the one workflow run id of their line 5, the only copies in this change
that do: each is byte-identical because its sha256 is the approved pin
(`04813918…`), so **redact** is not applied to them. Every source's sha256
equals its entry in the `SHA256SUMS` of the screen's durable copy, the orphan
branch `claude/handoff-ecs` at `067b8a10` (read in its local clone), which
also keeps what is not copied here.

- **The approved text and its approval.** `criteria.txt` (the approved
  criteria, written read-only at the launch) and `final/criteria_ecs_final.txt`
  (the same bytes, the text the approval request named), `APPROVED` (its
  `criteria_sha256`, the run log's line of the approval, the code tree digest
  and the five tools' sha256), `final/APPROVAL_REQUEST.txt` (the request,
  with V1 to V5's results and the open items its approval accepted),
  `final/diff_v3_to_final.txt` (the final text is draft v3 with only its
  PINS filled) and `SHA256SUMS.txt` (the screen's own sums of `final/` and
  `tools/`).
- **The design.** `FOR_THE_MAINTAINER.txt` (draft v2 for the maintainer:
  what it would answer and not, its cost and the decisions D1 to D12 and T1
  to T6 the maintainer took at 21:05 UTC on 2026-10-28),
  `v2/decisions_ecs_v2.txt` (each decision's options and reasons) and
  `survey_design.txt` (the design survey of the r10 ledgers, panels and
  probes: where the habit stood at the candidate starts, the entropy
  coefficient applied there, the power table).
- **The harness.** `tools/`: the five pinned tools, `ecs_common.py` (pins and
  constants), `run_ecs.py` (the continuation runner; copied as
  `run_ecs.py.gz`, since it reads `policy_loading._ensure_sb3` and
  `test_policy_loading.py` parses every Python file under `docs/`; `gunzip`
  restores it byte for byte), `read_ecs.py` (CHECK and the report-only
  readouts R1 to R4), `score_ecs.py` (the exact stratified test) and
  `run_waves_ecs.sh` (the wave runner), with their six test files in
  `tools/tests/`. `build_notes/LANE_H.txt` and `build_notes/LANE_R.txt` are
  the two builders' notes (LANE_H's section 3 holds the nine harness choices
  the approval accepted) and `build_notes/FIXES.txt` the fixer's record of the
  build's two reviews.
- **The runs.** `run_logs/decided.tsv` (each continuation's decision, all 24
  COUNTED with stop reason "end"), `run_logs/crash_ledger.tsv` (empty: no
  crash) and `run_logs/run_waves_ecs.log` (the wave runner's log, the only
  log the watcher read while the screen ran), and for each continuation
  `runs/<name>/run_meta.json` (its arguments, steps, seeds, code seal and
  stop reason) and `runs/<name>/check.json` (CHECK's result).
- **The reading.** `score/score.txt` (the scorer's output, STATUS final,
  run once at 12:15:29 UTC on 2026-10-29: the reading in its lines 53 to 85,
  then the report-only readouts per continuation) and `score/score.json`;
  for each continuation `score/read/<name>/read.json` (CHECK and the
  readouts R1 to R4, among them R2's per-actuator training-row shares and
  R3's end-trace commands, which `score.txt` prints only in summary),
  `m5.json` and `m5.txt` (the round's ledger rule M5 on its ledger).
- **The independent checks.** `check/stat/` (the statistic lens: every
  ledger's evaluation grid and parent rows re-derived with its own code, 30
  fields per continuation compared with the scorer's files, the exact test
  recomputed; `NOTES.txt`, its three scripts and their outputs) and
  `check/proc/` (the process lens: classes, pins, the code tree, the order
  of the files' births against the "all 24 decided" line, the printed
  reading; `NOTES.txt`, `scripts/` and the outputs). Both agree, with no
  blocking finding; the statistic lens's re-derivation reads each
  continuation's ledger, which is on the durable branch, not here.

Not copied: each continuation's ledger
(`runs/<name>/01_stance/eval_episodes.csv`, about 1.2 to 1.4 MB each) and its
other run files, the checkpoints, policy probes and training episodes; the end
traces (`traces/`); the per-continuation logs (`logs/<name>.*`); the readouts'
arrays (`score/read/<name>/r2_arrays.json`, about 0.3 MB each); the validation
runs, reviews, drafts v1 to v3 but the cited decisions file, the sealed code
copy and the scratch folders. The text outputs among them are on the durable
branch.
