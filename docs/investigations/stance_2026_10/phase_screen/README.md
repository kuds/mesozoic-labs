# The critic-phase screen (PS): from seed 42's r10 checkpoint at 2,097,152 steps (2026-10-31)

Evidence for decisions D-D60 and D-D62
([BEHAVIOR_RECIPES_PLAN.md](../../../BEHAVIOR_RECIPES_PLAN.md) §6.2) and for
§10 of [TREX_TAIL_SATURATION_HABIT_2026_10.md](../../TREX_TAIL_SATURATION_HABIT_2026_10.md)
(PS and VG there): the screen the maintainer chose at 18:30 UTC on
2026-10-30 ("Test the phase fix first (Recommended)") and confirmed as built
at 15:54:59 UTC on 2026-10-31 ("Confirm as built (Recommended)"), from its
build and criteria to its reading, final, the reading's two independent
checks and the report, with the separate report-only V − G and tail σ tool
the maintainer asked for in the same answer ("Build it during the runs
(Recommended)"). It continued `stance6_r10LTSB_s42` from its checkpoint at
2,097,152 steps, mid-rise, to 3,538,944 (175 updates, 1,441,792 steps, the
second critic-blind screen's window; 11 evaluations; the end pair is
`stage1_final`), in 4 control continuations and 4 `critic_phase` ones:
only the critic also sees the episode phase, `min(k, 200)/200`, a settled
flag and `0.98^max(0, 200 - k)` (k the episode step), carried in the three
command inputs (slots 60 to 62) that are always zero in stance; the actor is
unchanged and the critic still sees the `tail_3` angle. Paired training
seeds 2901, 2911, 2921 and 2931; the sealed code copy of the first
critic-blind screen, read only (code tree digest `331fe8fc…`, commit
`8737cd3d` of the never-merged run branch). The second critic-blind screen,
from the same state, is [`../bias_screen2/`](../bias_screen2/README.md).
Screen runs certify nothing, and nothing it ran is on `main`.

**The reading, final** (`score/score.txt`, `score_bs` 3.1, 20:19:27 UTC):
line 2, "REDUCED: GIVING THE CRITIC THE EPISODE PHASE (the tail_3 angle
still in its input; the actor unchanged) REDUCED THE ON-POLICY TAIL_3
COMMAND OVER THE LAST 8 UPDATES BEFORE E, FROM s42's 2,097,152 STATE (exact
one-sided permutation test, p <= 0.05)". The primary, the on-policy `tail_3`
command over the last 8 updates before E, against B_S = +0.12660 at S:
control +0.49357, +0.58592, +0.66068 and +0.64649, critic_phase +0.06767,
+0.04724, +0.04290 and +0.06832 (seeds 2901 to 2931); the control median
+0.61621, a rise of +0.48961 against the pre-registered 0.20 (the parent's
own rise over the window +0.57878); the critic_phase median +0.05745, below
B_S; p 1/70, the smallest attainable with 4 per arm (filled and as
counted). The pre-registered secondaries, which never change the line:
SEC-CMD (the ledger's `tail_3` command at E) SUPPORTED, p 1/70; SEC-MARGIN
(the margin to the rail at E, b* = 1 - G - c0) SUPPORTED, p 1/70; SEC-MECH
(the critic's own `tail_3` slope at E) SUPPORTED, p 1/70, each surviving
Bonferroni over the three; the guard SEC-TAIL2 (the on-policy `tail_2`
command, the phase arm higher) SUPPORTED, p 2/70, a warning of small size.
All 8 COUNTED, none missing; the manipulation held on all 175 rollouts of
every continuation. Both independent checks agree (`check/stat/`,
`check/proc/`), and the report (`REPORT.md`) applies the statistics check's
spread caveat.

[SOURCES.txt](../SOURCES.txt) lists each file's source, its sha256 and the
changes the copy made; [SHA256SUMS](../SHA256SUMS) pins them (every file of
the folder but this README). Each copy's source is the file at the same path
below `work/phase_screen/` (the session's work folder, `/home/user/work`, the
run log's W), but `vg_tool/` and `vg_run/`, whose sources are below
`work/vg_tool/` and `work/vg_run/`, and `run_logs/`,
`check_build/run_logs/` and `review/<lens>/run_logs/`, whose sources are the
`logs/` folders at those places (the repository's `.gitignore` ignores every
directory named `logs`). Every source's sha256 equals its entry in the
SHA256SUMS of the screen's durable copy, the orphan branch
`claude/handoff-phase-screen` at `dd4f0d32` (its stage, read locally),
never merged.

- **`criteria.txt`**, **`APPROVED`**, **`SCREEN.json`** and
  **`SHA256SUMS.txt`**: the approved criteria (sha256 `9875039f…`,
  APPROVED's `criteria_sha256`), the approval record (sha256 `067d97d2…`,
  the run log's pin, written at 15:56:13 UTC, 45 s before the launch at
  15:56:58 UTC; it pins the criteria, `SCREEN.json` (`7d6d368a…`), the code
  tree digest and the six tools), the queue and the builder's sums.
- **`REPORT.md`**: the result as reported to the maintainer at 20:41 UTC,
  with the V − G table, what it does not show and the process notes.
- **`NOTES.txt`**: the harness builder's notes (the second critic-blind
  screen's tools re-pointed and the `critic_phase` arm built: the phase fed
  into the critic's input columns of the three slots, the actor's columns
  for them kept bitwise unchanged with zero Adam moments, the slots
  self-mirrored and their normaliser statistics reseeded and frozen; the
  validations V1 to V5, with no optimizer step but V5's one Adam step on a
  throwaway in-memory copy).
- **`tools/`**: the six pinned tools (`run_bs.py`, gzipped, `read_bs.py`,
  `run_waves_bs.sh`, `bs_common.py`, `bs_hooks.py`, `score_bs.py`), their
  five test files and two helpers.
- **`check_build/`**: the independent build check (its notes, scripts and
  outputs: the diff from the second screen's tools, the arm's actor and
  critic checks on a validation pair, the control's equality with the
  second screen's control, the reader, the runner's refusals, the tests).
- **`criteria_work/`**: the designer's notes (`DESIGN_NOTES.txt`: B_S, the
  gate, the power, the secondaries, the seeds), the scorer adapter's notes
  (`ADAPT_NOTES.txt`, the edits A1 to A26 that reconciled the draft with
  the harness) and logs (`adapt_logs/`), their command log and scripts and
  the drafts' sums.
- **`review/`**: the design review (`design/`, 0 blocking, D1 to D4 minor),
  the process review (`process/`, one blocking finding, P-B1: the training
  library's TensorBoard events went to a folder outside the screen's), the
  fixer's record (`FIXES.txt`, with `fixer/`: 32 findings, P-B1 fixed and
  re-validated) and the re-review of the fixed version (`rereview/`: READY
  TO PIN, 0 blocking), with `reviewed_sha256.txt`.
- **`runs/<name>/`**: each COUNTED continuation's `run_meta.json` and
  `check.json` (the reader's CHECK: passed); **`runs/<name>_crashed1/`** and
  **`_crashed2/`**: the `run_meta.json` alone of the second wave's attempts
  killed by the two container restarts (about 18:07 and 18:53 UTC), each
  recorded as a crash not counted and rerun from the start pair with the
  same seed under the runner's rules; nothing else of them was read.
- **`run_logs/`**: the runner's logs (`run_waves_bs.log` and its `.out`
  copy, `decided.tsv`: all 8 COUNTED, `crash_ledger.tsv`: the 8 crashes not
  counted, `ALL_DECIDED`, the access-time snapshots at each exit and at
  ALL_DECIDED, the exit record, the runner's stderr, each counted attempt's
  log and each continuation's check and read logs) and the build's and the
  pin's records (the build's command log and explained hunks, the tools'
  diff from the second screen's, the pre-launch plan, preflights and score,
  the tools' sums as copied and after the launch, the code copy's
  verification at start and end, and the scorer's run).
- **`reads/<name>/`**: each continuation's read, as `markers.txt` and
  `m5.txt`.
- **`score/`**: `score.txt` (sha256 `2d9793ef…`) and `score.json`
  (`bda79018…`), the values the run log gives.
- **`check/stat/`** and **`check/proc/`**: the two independent checks: the
  statistic (931 comparisons with the score, 0 differences beyond float
  round-off; the primaries as exact rationals of the probe rows; the spread
  caveat, SD ratio 5.69) and the process (the pins, APPROVED's timing, the
  run handling of the two restarts, outcome-blindness from the access-time
  snapshots and the transcripts, as far as they can show; findings M1 and M2,
  both minor).
- **`vg_tool/`** and **`vg_run/`**: the separate report-only V − G and tail
  σ tool (`SPEC.txt`, `vg_sigma.py` 1.1, its tests, notes, fixes and
  command log; pinned in the run log at 17:43:18 UTC, before the screen was
  decided) and its run after the reading (`out/vg_sigma.txt`, `.json`,
  `run_info.json`, and `run.out`): never a line or a secondary.

Every copy is byte-identical to its source, absolute session paths
included, but for two existing kinds of change. **redact**: in
`check/proc/NOTES.txt`, `cmdlog.txt`, `SHA256SUMS.txt` and
`out/sub_summary.out`, and in `criteria_work/CMDLOG.txt` and
`DESIGN_NOTES.txt`, 5 workflow run ids (3 distinct), 6 sub-agent ids (3
distinct, in the names of three transcript files not kept) and 3
background-job or tool-result ids of the session's task runner (2
distinct), each by
`[id removed]` (a bare workflow id by `workflow [id removed]`); `APPROVED`
keeps the one workflow run id it holds, since its sha256 is the run log's
pin. **gzip**: `tools/run_bs.py` is copied as `tools/run_bs.py.gz` (it reads
`policy_loading._ensure_sb3`, and `test_policy_loading.py` parses every
Python file under `docs/`); gunzip restores it byte for byte, with the
sha256 APPROVED and `SHA256SUMS.txt` pin. No other copy held a session tool
id, and none holds a model identifier, an email address or a commit
trailer.

Not kept (all of it also on the durable branch, but the code copy, the
checkpoints and other npz, zip and pkl files, the training episodes, the
validations' ledgers and the temporary files): each continuation's run
folder but its `run_meta.json` and `check.json` (the ledger, the training
episodes, the policy probe, the arm evidence per rollout and the trainer's
other run records) and each killed attempt's folder but its `run_meta.json`;
the reads' JSON (their `.txt` renderings are kept); the validations
(`validate/`); the drafts (`criteria_work/criteria_v1.txt`,
`SCREEN_v1.json`), the review's copy of the file it reviewed
(`review/criteria_reviewed_v1.txt`), the fixer's copies of the files before
its edits (`review/fixer/orig/`) and the design review's 22 mutated copies of
the scorer (`review/design/work/mutants/`; their run logs are kept); the
validation and synthetic roots the build check, the dry runs and the
re-review built (`check_build/ctrl_root/`, `runner_root/`,
`criteria_work/dry_*`, `review/fixer/dry/`, `review/rereview/dry/`,
`synth/`, `pins_root/`); the process check's extracts of the session's own
transcripts (`check/proc/out/tx_*.txt`, `tx_calls_*.jsonl` and three
`sub_agent-*.txt`, which hold the session's internal tool, task and agent
ids; `tx_extract.out`, `tx_rules.out`, `sub_classify.out` and
`sub_summary.out` hold what the check found in them); 105 empty files
(91 empty error logs, empty outputs of commands that printed nothing, and
empty marker files); the runner's process-table snapshots
(`logs/proctable.log`), which list the container's own processes; and 20
files that list the container's files outside the screen's folder, with
their searches' error logs (`*outside*` in `logs/`, `check_build/logs/`,
`criteria_work/` and `review/`; `review/fixer/run_logs/outside_ps_record.txt`,
the fixer's note on them, is kept). The scripts read the session's paths
they name and are kept as records of what was run, not to be run from here.

To re-check the approved text and the tools: `sha256sum criteria.txt
SCREEN.json APPROVED` against `APPROVED`'s `criteria_sha256` and
`screen_sha256` and the run log's `067d97d2…`; in a copy of this folder,
`gunzip tools/run_bs.py.gz`, then `sha256sum -c --ignore-missing
SHA256SUMS.txt` (the builder's sums also list files not kept) and each
tool's sha256 against `APPROVED`'s `tool_sha256`.

Kept as their sources have them, so to leave out of a manual pre-commit run:
`check/stat/out/manip_check.json`, `spread.json` and `stat_check.json` and
`review/process/work/pin_check.json` have no final newline;
`criteria_work/rr_findings.out` ends in a blank line; and these 22 files have
lines ending in spaces (in the diffs, blank lines that keep the space after
their diff marker): `check/proc/out/sub_classify.out` (2) and `tx_rules.out`
(3); `check/stat/out/tree_check_meta_phase2901.txt` (1) and
`tree_meta_control2901.txt` (1); `check_build/my_tools_vs_bs2.diff` (116)
and `run_logs/runner_refusals.out` (13); `criteria_work/adapt_logs/install.out`
(3) and `plan.out` (1); `review/fixer/run_logs/inspect_diffs.out` (14),
`plan.out` (1), `preflight_all_2.out` (8), `pytest_score_1.out` (1) and
`val/driver.log` (2); `review/process/run_logs/runner_cases.out` (11);
`review/rereview/run_logs/diff_run_bs.out` (9), `diff_score_bs.out` (9),
`diff_test_run_bs.out` (2) and `diff_test_score_bs.out` (11);
`run_logs/prelaunch/plan.out` (1), `score_run.out` (6) and
`tools_vs_bs2.diff` (116); and `score/score.txt` (6). `git diff --check`
lists each.
