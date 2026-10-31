# The first critic-blind screen (BS): from seed 42's r10 checkpoint at 1,048,576 steps (2026-10-30)

Evidence for decision D-D56 (2) (b)
([BEHAVIOR_RECIPES_PLAN.md](../../../BEHAVIOR_RECIPES_PLAN.md) §6.2) and for
§9 of [TREX_TAIL_SATURATION_HABIT_2026_10.md](../../TREX_TAIL_SATURATION_HABIT_2026_10.md)
(BS there): the one scratch CPU training screen the maintainer authorised at
00:45 UTC on 2026-10-30, from its criteria to its reading, final, and the
reading's two independent checks. It continued `stance6_r10LTSB_s42` (the
first r10 stance round's seed 42, whose results are
[`../stance6_results/`](../stance6_results/README.md)) from its checkpoint at
1,048,576 steps, before its tail rose, to 2,490,368 (175 updates, 1,441,792
steps), in 4 control continuations and 4 with `critic_blind_tail3` (the
normalised `tail_3` angle, observation 6, set to 0 at the value network's
input only; the actor still sees it), paired training seeds 2101, 2111, 2121
and 2131, trained from the sealed code copy of the never-merged run branch at
`8737cd3d` (code tree digest `331fe8fc…`). The session chose the
intervention under the authorisation (the run log's entry timed 04:22 UTC).
Screen runs certify nothing, and nothing it ran is on `main`.

**The reading, final** (`score/score.txt`, `score_bs` 1.1, 09:03 UTC): line
1, "NOT INFORMATIVE: THE CONTROL ARM DID NOT REPRODUCE THE EARLY RISE OF THE
ON-POLICY TAIL_3 COMMAND (its median primary rose by less than 0.10 from the
value at S, or no control primary was read); no reading about the critic is
made". The primary, the on-policy `tail_3` command (the policy probe's
`rollout_mu14_mean`) over the last 8 updates before E, against B_S = -0.03157
at S: control -0.03102, +0.02567, +0.09486 and +0.28273, critic blind
+0.02868, -0.02845, +0.03689 and +0.01563 (seeds 2101 to 2131); the control
median +0.06027, a rise of +0.09184 against the pre-registered 0.10. Report
only: the arm means +0.09306 and +0.01319, the exact one-sided permutation p
13/70 and the paired sign-flip 4/16; all 8 COUNTED, none missing; the
manipulation held on all 175 rollouts of every continuation. Both
independent checks agree (`check/stat/`, `check/proc/`).

[SOURCES.txt](../SOURCES.txt) lists each file's source, its sha256 and the
changes the copy made; [SHA256SUMS](../SHA256SUMS) pins them (every file of
the folder but this README). Each copy's source is the file at the same path
below `scratchpad2/r10round/bias_screen/`, but `run_logs/`, whose source is
that folder's `logs/` (the repository's `.gitignore` ignores every directory
named `logs`). Every source's sha256 equals its entry in the SHA256SUMS of
the screen's durable copy, the orphan branch `claude/handoff-bias-screen` at
`cfed2c70` (read in its local clone).

- **`criteria.txt`**, **`APPROVED`**, **`SCREEN.json`**,
  **`SCREEN.template.json`** and **`SHA256SUMS.txt`**: the approved criteria
  (v2, sha256 `9c490cc6…`, APPROVED's `criteria_sha256`), the approval record
  (sha256 `7bf25afd…`, written 06:00:05 UTC, 14 s before the launch; it pins
  the criteria, `SCREEN.json` (`a4eca4dc…`), the code tree digest and the six
  tools), the queue and its template, and the builder's sums.
- **`NOTES.txt`** and **`NOTES_criteria.txt`**: the build notes of the
  scaffold (the harness, the arm hook with four arms behind flags, the
  reader, V1 to V3 with no optimizer step, 72 tests) and of the criteria and
  scorer (the primary, the gate's power, the test).
- **`tools/`**: the six pinned tools (`run_bs.py`, gzipped, `read_bs.py`,
  `run_waves_bs.sh`, `bs_common.py`, `bs_hooks.py`, `score_bs.py`), their
  five test files and two helpers (`check_inputs.py`, `verify_copy.py`).
- **`review/`**: the independent review of criteria v1 and the scorer's v1
  (`NOTES.txt`, its scripts and outputs; 9 findings, none blocking) and the
  fixer's record (`FIXES.txt`, with `fixer/`: its checks, the diffs v1 to v2
  of the criteria, the scorer and its tests, and a scorer run on the root
  before any continuation); the fixed version was not re-reviewed (the
  process check's P2).
- **`scratch/`**: the criteria's design notes (`notes_design.txt`,
  `notes_top.txt`, `explore_policy.py`) and the criteria writer's checks in
  `criteria_work/` (the gate's power, the entropy screen's spread, the
  scorer's dry run and mutants, the extra-key check of APPROVED, a
  preflight, the queue plan and a scorer run on the root before any
  continuation).
- **`runs/<name>/`**: each continuation's `run_meta.json` (its pins, arm,
  seed, steps, exit and stop reason) and `check.json` (the reader's CHECK of
  its run records, arm evidence and probe rows: passed).
- **`run_logs/`**: the runner's logs: `run_waves_bs.log` (and its `.out`
  copy), `decided.tsv` (all 8 COUNTED), the empty crash ledger,
  `ALL_DECIDED`, the exit record, the runner's stderr and each
  continuation's attempt, check and read logs.
- **`reads/<name>/`**: each continuation's read, as `markers.txt` (the
  primary's window, the ledger's `tail_3` command, and at each saved pair the
  loop gain G, the offset c0, the margin b* and the critic's slope on the
  tail angle) and `m5.txt` (the ledger's flags, M5), and the parent's
  (`reads/parent_s42/`).
- **`score/`**: the scorer's reading, `score.txt` and `score.json` (sha256
  `e5faa4a3…`, the value the run log gives).
- **`check/stat/`** and **`check/proc/`**: the two independent checks of the
  reading, the statistic (every primary, the median, the rise +0.091843,
  exact 3155692807/34359738368, p 13/70 and the sign-flip 4/16 recomputed
  with its own code; the run log's report-only claims, with one wording
  correction, STAT-1) and the process (the pins, the timing of APPROVED, the
  arm evidence per rollout, outcome-blindness from the file access times, as
  far as they can show (relatime cannot exclude a re-read of the check's
  inputs, P4), the authorisation's limits). The checkers' summaries are in
  the run log's entry timed 09:13 UTC (line 780, the copy's line 21).

Every copy is byte-identical to its source, absolute session-scratch paths
included (no **prefix**, **prefix2** or **checkout**, as in `../stance6/`),
but for two existing kinds of change. **redact**: one workflow run id in each
of `NOTES_criteria.txt`, `review/FIXES.txt` and `review/NOTES.txt` and two in
`review/fixer/criteria_v1_to_v2.diff`, each after the word "workflow", by
`[id removed]`; `criteria.txt` keeps the one workflow run id of its line 4,
since its sha256 is the approved pin (as `../ecs_results/criteria.txt` keeps
its own). **gzip**: `tools/run_bs.py` is copied as `tools/run_bs.py.gz` (it
reads `policy_loading._ensure_sb3`, and `test_policy_loading.py` parses
every Python file under `docs/`); gunzip restores it byte for byte, with the
sha256 APPROVED and `SHA256SUMS.txt` pin. `run_logs/crash_ledger.tsv` is
empty, as its source is. No other copy held a session tool id, and none
holds a model identifier, an email address or a commit trailer.

Not kept (all of it also on the durable branch, but the code copy, the
checkpoints and other npz, zip and pkl files, the training episodes, the
validations' ledgers, the temporary files and the empty lock file): the
sealed code copy (`code/`: 1,406 files in its tree digest and its seal,
`.r10m_seal.json`); each continuation's run folder but its `run_meta.json`
and `check.json` (the checkpoints, the ledger `eval_episodes.csv`, the
training episodes, the policy probe, the arm evidence per rollout
`arm_evidence.jsonl` and the trainer's other run records); the reads' JSON
(`markers.json`, `m5.json`; their `.txt` renderings are kept), the reader's
loop cache (`reads/_cache/`) and the parent's ledger cut
(`reads/parent_s42/eval_episodes_le_E.csv`); the validations (`validate/`:
the V1 to V3 runs, their scripts and logs); the fixer's copies of the
pre-fix files (`review/fixer/orig/`) and the two pre-edit test copies
(`scratch/criteria_work/*.orig`); the synthetic dry-run root
(`scratch/criteria_work/dryrun_root/`); the temporary files (`tmp/`); the
runner's lock file; and its process-table snapshots (`logs/proctable.log`),
which list the container's own processes. The scripts read the session's
scratch paths they name and are kept as records of what was run, not to be
run from here.

To re-check the approved text and the tools: `sha256sum criteria.txt
SCREEN.json` against `APPROVED` (`criteria_sha256`, `screen_sha256`); in a
copy of this folder, `gunzip tools/run_bs.py.gz`, then `sha256sum -c
--ignore-missing SHA256SUMS.txt` (the builder's sums also list the
validations, which are not kept) and each tool's sha256 against `APPROVED`'s
`tool_sha256`.

Kept as their sources have them, so to leave out of a manual pre-commit run:
`review/fixer/criteria_v1_to_v2.diff` (7 lines), `score_bs_1.0_to_1.1.diff`
(8) and `test_score_bs.diff` (6), `score/score.txt` (5) and
`scratch/criteria_work/waves_plan.out` (1) have lines ending in spaces (in
the diffs, the single space of their blank context lines).
