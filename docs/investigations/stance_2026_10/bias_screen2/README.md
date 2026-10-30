# The second critic-blind screen (BS2): from seed 42's r10 checkpoint at 2,097,152 steps (2026-10-30)

Evidence for decision D-D57
([BEHAVIOR_RECIPES_PLAN.md](../../../BEHAVIOR_RECIPES_PLAN.md) §6.2) and for
§9 of [TREX_TAIL_SATURATION_HABIT_2026_10.md](../../TREX_TAIL_SATURATION_HABIT_2026_10.md)
(BS2 there): the screen the maintainer chose at 10:23 UTC on 2026-10-30 ("Go
with option a", the overnight report's option A) and launched with its 0.20
gate at 12:53 UTC ("Launch with 0.20 (Recommended)"), from its build and
criteria to its reading, final, and the reading's two independent checks.
It continued `stance6_r10LTSB_s42` from its checkpoint at 2,097,152 steps,
mid-rise, to 3,538,944 (175 updates, 1,441,792 steps, 11 evaluations; the
end pair is `stage1_final`), in 4 control continuations and 4 with
`critic_blind_tail3` (the normalised `tail_3` angle, observation 6, set to 0
at the value network's input only; the actor still sees it), paired training
seeds 2501, 2511, 2521 and 2531, on the first screen's sealed code copy, read
only (code tree digest `331fe8fc…`). The first screen is
[`../bias_screen/`](../bias_screen/README.md). Screen runs certify nothing,
and nothing it ran is on `main`.

**The reading, final** (`score/score.txt`, `score_bs` 2.1, 15:57 UTC): line
2, "REDUCED: BLINDING THE CRITIC TO THE TAIL_3 ANGLE (observation 6) REDUCED
THE ON-POLICY TAIL_3 COMMAND OVER THE LAST 8 UPDATES BEFORE E, FROM s42's
2,097,152 STATE (exact one-sided permutation test, p <= 0.05)". The primary,
the on-policy `tail_3` command over the last 8 updates before E, against B_S
= +0.12660 at S: control +0.67001, +0.79227, +0.51081 and +0.41938, critic
blind +0.14283, +0.12485, +0.13810 and +0.06516 (seeds 2501 to 2531); the
control median +0.59041, a rise of +0.46382 against the pre-registered 0.20
(the parent's own rise over the window +0.57878); p 1/70, the smallest
attainable with 4 per arm (filled and as counted); the arms' spreads differ
(SD 0.166 against 0.036), and had blinding only narrowed the spread a
separation this complete would occur about 3% of the time, not 1.4%
(`REPORT.md`'s spread caveat; the pre-registered p decides). The
pre-registered secondaries, which never change the line: SEC-CMD (the
ledger's `tail_3` command at E) SUPPORTED, p 1/70; SEC-MARGIN (the margin to
the rail at E, b* = 1 - G - c0) SUPPORTED, p 1/70; SEC-G (the end loop gain,
the first screen's post hoc pattern tested out of sample) NOT SUPPORTED,
70/70. All 8 COUNTED, none missing; the manipulation held on all 175
rollouts of every continuation. Both independent checks agree
(`check/stat/`, `check/proc/`), with the wording corrections that
`REPORT.md` applies.

[SOURCES.txt](../SOURCES.txt) lists each file's source, its sha256 and the
changes the copy made; [SHA256SUMS](../SHA256SUMS) pins them (every file of
the folder but this README). Each copy's source is the file at the same path
below `scratchpad2/r10round/bias_screen2/`, but `run_logs/`,
`check_build/run_logs/`, `review/fix/run_logs/`,
`review/process/run_logs/` and `review/rereview/run_logs/`, whose sources
are the `logs/` folders at those places (the repository's `.gitignore`
ignores every directory named `logs`). Every source's sha256 equals its
entry in the SHA256SUMS of the screen's durable copy, the orphan branch
`claude/handoff-bias-screen2` at `2d241410` (read in its local clone), but
`review/process/work/copyroot_sha256.txt`, which that copy's name filter
left out.

- **`criteria.txt`**, **`APPROVED`**, **`SCREEN.json`** and
  **`SHA256SUMS.txt`**: the approved criteria (sha256 `291ff457…`, APPROVED's
  `criteria_sha256`), the approval record (sha256 `184f9ab9…`, the run log's
  pin, written 12:53:50 UTC, 46 s before the first start; it pins the
  criteria, `SCREEN.json` (`255d0ba4…`), the code tree digest and the six
  tools), the queue and the builder's sums.
- **`REPORT.md`**: the result as reported to the maintainer at 16:22 UTC,
  with the checkers' wording corrections S-2 to S-7 applied, and what it does
  not show.
- **`NOTES.txt`**: the harness builder's notes (the first screen's tools
  re-pointed at 2,097,152 in 38 explained hunks, the window, the start pair,
  V1 to V3 with no optimizer step, the reader's cache key fixed (the first
  screen's process finding P3)).
- **`tools/`**: the six pinned tools (`run_bs.py`, gzipped, `read_bs.py`,
  `run_waves_bs.sh`, `bs_common.py`, `bs_hooks.py`, `score_bs.py`), their
  five test files and two helpers.
- **`check_build/`**: the independent build check (its notes, scripts and
  outputs: the sums, the diff from the first screen's tools, the tests, its
  own start-pair, window and V2 checks, the runner's refusals; 0 blocking).
- **`criteria_work/`**: the designer's notes (`DESIGN_NOTES.txt`: B_S, the
  gate, the power, the secondaries, the seeds), the scorer adapter's notes
  (`ADAPT_NOTES.txt`), their command log and scripts (the parent's window and
  extras, the power, the first screen's secondaries read post hoc) and the
  drafts' sums.
- **`review/`**: the design and statistics review (`design/`) and the harness
  and process review (`process/`), 0 blocking and 32 findings in all; the
  fixer's record (`FIXES.txt`, with `fix/`: 31 findings confirmed, 25 fixed
  and 6 not needed or optional, 1 rejected with its evidence; its edit
  scripts, dry runs and re-run checks); and the re-review of the fixed
  version (`rereview/`: READY TO PIN, 0 blocking, one minor finding, RR-1,
  handled by procedure), with `reviewed_sha256.txt`.
- **`runs/<name>/`**: each continuation's `run_meta.json` and `check.json`
  (the reader's CHECK: passed).
- **`run_logs/`**: the runner's logs (`run_waves_bs.log` and its `.out`
  copy, `decided.tsv`: all 8 COUNTED, the empty crash ledger, `ALL_DECIDED`,
  the access-time snapshot taken at ALL_DECIDED before the full reads
  (`atimes_at_ALL_DECIDED.tsv`), the exit record, the runner's stderr, each
  continuation's attempt, check and read logs) and the build's and the
  pin's records (the build's command log and the explained diff from the
  first screen's tools, the input checks, the pre-launch plan, queue,
  preflights and score, the tools' sums as copied and after the launch, the
  code copy's verification at start and end, and the scorer's stdout).
- **`reads/<name>/`**: each continuation's read, as `markers.txt` and
  `m5.txt`.
- **`score/`**: `score.txt` (sha256 `71af69cb…`) and `score.json`
  (`87d07a4e…`), the values the run log gives.
- **`check/stat/`** and **`check/proc/`**: the two independent checks: the
  statistic (104 comparisons with `score.json`, 0 differences; the primaries
  as exact rationals of the probe rows, B_S 135930855/2^30, the rise
  498021527/2^30, p 1/70 with the next statistic +0.34211 against the
  observed +0.48039; the report's claims, with the corrections S-2 to S-7)
  and the process (the pins, APPROVED's timing, the arm evidence per rollout,
  outcome-blindness from the access-time snapshot and its own scan, as far as
  they can show (they cannot see a read before each decision, I1), the
  limits, the remote's refs during the screen; its other findings for
  information only). The checkers' summaries are in the run log's entry timed
  16:21 UTC (line 796, the copy's line 37).

Every copy is byte-identical to its source, absolute session-scratch paths
included (no **prefix**, **prefix2** or **checkout**), but for two existing
kinds of change. **redact**: one workflow run id in
`criteria_work/DESIGN_NOTES.txt` (after the word "workflow"), and in the
process check's transcript reads (`check/proc/work/transcript_*.out`, four
files) 3 workflow run ids (2 distinct, each inside a path or a file name), 1
scheduled-task id and 8 background-job ids of the session's task runner (4
distinct, nine characters), each by `[id removed]`; `criteria.txt` and
`APPROVED` keep the one workflow run id each holds (the criteria's line 5,
APPROVED's line 13), since their sha256 are the approved pin and the run
log's (as `../ecs_results/criteria.txt` keeps its own). **gzip**:
`tools/run_bs.py` is copied as `tools/run_bs.py.gz` (it reads
`policy_loading._ensure_sb3`); gunzip restores it byte for byte, with the
sha256 APPROVED and `SHA256SUMS.txt` pin. `run_logs/crash_ledger.tsv` is
empty, as its source is. No other copy held a session tool id, and none
holds a model identifier, an email address or a commit trailer.

Not kept (all of it also on the durable branch, but the code copy, the
checkpoints and other npz, zip and pkl files, the training episodes, the
validations' ledgers, the dry-run roots `dryrun_root/` and `copyroot/`, the
temporary files and the empty lock file): each continuation's run folder but
its `run_meta.json` and `check.json` (the checkpoints, the ledger, the
training episodes, the policy probe, the arm evidence per rollout and the
trainer's other run records); the reads' JSON (their `.txt` renderings are
kept) and the reader's loop cache; the validations (`validate/`); the drafts
(`criteria_work/criteria_v1.txt`, `SCREEN_v1.json`) and the reviews' copies
of the files they reviewed (`review/criteria_reviewed_v1.txt`,
`review/rereview/copies/`); the synthetic roots the dry runs, mutation
checks and re-review built (`dryrun_root/`, `copyroot/`, `killroots/`,
`rr_roots/`, `root_prelaunch_pinned/`), the fixer's pytest temporaries and
the trainer's temporary files (`tmp/`); the process check's raw access-time
scan (`check/proc/my_atime_scan_before_reading.tsv`, 1,073,346 bytes, over
the repository's 500 KB limit for an added file;
`check/proc/work/p4_audit.py` reads it, and `p4_audit.out` holds what it
found); 47 empty files (empty error logs, empty outputs of searches that
found nothing, and empty markers); the runner's lock file and its
process-table snapshots (`logs/proctable.log`), which list the container's
own processes; and two listings of the container's files outside the
session's folders (`logs/outside_bs2_newer_than_marker.txt`,
`review/fix/logs/tmp_top_newer.txt`). The scripts read the session's scratch
paths they name and are kept as records of what was run, not to be run from
here.

To re-check the approved text and the tools: `sha256sum criteria.txt
SCREEN.json APPROVED` against `APPROVED`'s `criteria_sha256` and
`screen_sha256` and the run log's `184f9ab9…`; in a copy of this folder,
`gunzip tools/run_bs.py.gz`, then `sha256sum -c --ignore-missing
SHA256SUMS.txt` (the builder's sums also list files not kept) and each
tool's sha256 against `APPROVED`'s `tool_sha256`.

Kept as their sources have them, so to leave out of a manual pre-commit run:
`check/proc/work/gh_activity_day.json` and `review/process/work/pin_check.json`
have no final newline, and these 27 files have lines ending in spaces (in the
diffs, blank lines that keep the space after their diff marker):
`run_logs/tools_vs_bs.diff` (45 lines), `prelaunch_plan.out` (1) and
`score_stdout.txt` (6); `score/score.txt` (6);
`check/proc/work/transcript_calls_after_all_decided.out`,
`transcript_calls_during_screen.out` and `transcript_p1_times.out` (1 each);
`check_build/run_logs/my_tools.diff` (19) and `runner_refusal_mine.out` (1);
`check_build/work/d_test_read.txt` (8) and `d_test_run.txt` (12);
`review/design/work/rd_mutants.out` (2);
`review/fix/run_logs/criteria_v1_to_fixed.diff` (6),
`dryrun_parent_as_all.stdout` (8), `dryrun_synthetic_E.stdout` (8),
`plan_real.out` (1) and `pytest_score_1.out` (1);
`review/process/run_logs/plan_real.out` (1) and `runner_cases.out` (6);
`review/rereview/run_logs/dryrun_parent_as_all.stdout` (8),
`dryrun_synthetic_E.stdout` (8), `plan_real.out` (1), `score_rr_A.stdout` (8),
`score_rr_B.stdout` (7), `score_rr_C.stdout` (8) and `score_rr_D.stdout` (8);
and `review/rereview/work/diff_runner.txt` (3). `git diff --check` lists each.
