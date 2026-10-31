# The overnight no-training probes of the T. rex tail habit (2026-10-30)

Evidence for decision D-D56
([BEHAVIOR_RECIPES_PLAN.md](../../../BEHAVIOR_RECIPES_PLAN.md) §6.2) and for
§9 of [TREX_TAIL_SATURATION_HABIT_2026_10.md](../../TREX_TAIL_SATURATION_HABIT_2026_10.md),
where each folder's short name is given below. Four no-training probes of
the night of 2026-10-29 to 2026-10-30, each with an adversarial verifier,
on the first r10 stance round's local checkpoints (`stance6_r10LTSB_s42`,
`_s60` and `_s80`), the 24 entropy-screen continuations of
[`../ecs_results/`](../ecs_results/README.md) and, for the held-out test,
the r9 stance5 runs' records. Simulation, network evaluation and records
only, on physics r10 with the sealed code copy of the never-merged run
branch at `8737cd3d` (code tree digest `331fe8fc…`: the entropy screen's
copy, the same tree both critic-blind screens trained from), and on the r9
runs' records; exploratory and report only (the two follow-ups
pre-registered their rules, hashed before data).
Nothing here changes a gate, reward, plant or criterion, and nothing here is
a gate reading. The run log of the night and the overnight report that
summarised it for the maintainer are in
[`../critic_blind_results/`](../critic_blind_results/README.md).

[SOURCES.txt](../SOURCES.txt) lists each file's source, its sha256 and the
changes the copy made; [SHA256SUMS](../SHA256SUMS) pins them (every file of
the folder but this README). Each folder here is the folder of the same name
under `scratchpad2/com_revisit/notes/`. Every copy is byte-identical to its
source, absolute session-scratch paths included, but
`warning_r9/wr9_apply.py`, kept gzipped as `warning_r9/wr9_apply.py.gz`
(**gzip**: it uses a nested f-string with the same quotes, which Python 3.12
allows and Python 3.11 cannot parse, and `test_policy_loading.py` parses
every Python file under `docs/` on each Python version CI runs; gunzip
restores it byte for byte); none held a session tool id. Every source's
sha256 equals its entry in the SHA256SUMS of the first
screen's durable copy (the orphan branch `claude/handoff-bias-screen` at
`cfed2c70`, its `overnight_probes/`, read in its local clone).

- **`early_bias/`** (EB) and **`verify_early_bias/`** (VEB): the early-bias
  probe of D-D56 (1), asked for at 00:24 UTC. At the r10 local checkpoints
  (s42 and s80 at 1,048,576, 2,097,152 and 3,145,728 steps, s42 also at
  3,670,016, s60 from 5,242,880 to 8,388,608) one PPO update is rebuilt
  (rollout, GAE, the full mirror-symmetric PPO loss) with no optimizer step,
  and each loss term's first-order push on the deterministic `tail_3`
  command is split by reward term, state region and noise carrier and set
  against the weights' actual change to the next checkpoint. The verifier
  rebuilt it with its own code and fetched the 1,310,720 checkpoints of s42
  and s80 read-only from Drive. The verified readings: only the
  policy-gradient term moves the tail mean (the entropy and value terms
  exactly 0, the mirror term small); at 1,310,720, about 0.1M steps after
  s42's onset, s42 already has an upward push on `tail_3` carried by its own
  exploration noise (+0.00115 ± 0.00032 per epoch, t 3.6) and s80 none; the
  critic's bootstrapped value as its source is supported from 2,097,152 on
  and unresolved at the onset. EB has no notes file: `eb_summary.txt` and
  `eb_aggregate.txt` are its tables, `eb_update.py` the rebuilt update,
  `eb_actual.txt`, `eb_realized.txt`, `eb_realized_s60early.txt` and
  `eb_proxy.txt` its checks against the actual change and of the
  episode-phase reading; VEB's tables are `vb_aggregate.txt`,
  `vb_realized.txt` and `vb_bank_weights.txt`. Both reported to the session,
  and the run log's entries timed 03:45 and 04:21 UTC (lines 770 and 771,
  the copy's 11 and 12) hold their findings.
- **`early_warning/`** (EW) and **`verify_early_warning/`** (VEW): the
  early-warning curve of D-D56 (2) (a): the tail loop's gain G, its offset
  c0, fixed point and posture slope at every saved checkpoint of the three
  r10 runs and the 24 continuations, lined up against the ledgers' events,
  as candidate report-only telemetry. The verified reading: G turns positive
  one checkpoint before the first tail over-bar episode in s42 and s60, but
  late s80 reaches the same loop state with no over-bar episode, so no
  loop-only rule separates them with any margin; the record's early G values
  came from a sweep that reaches the normaliser's clip (EW `NOTES.txt` §5,
  VEW `vew_clip.txt`). `NOTES.txt`, `ew_analyze.txt` and `ew_series.csv`
  are EW's notes and tables; `vew_analyze.txt` and `vew_clip.txt` VEW's.
- **`warning_r9/`** (WR9), **`warning_margin/`** (WM) and
  **`verify_warning_followups/`** (VWF): the two follow-ups of the curve,
  launched at 02:42 UTC, and their verifier. WR9 tests, out of sample on the
  five r9 stance5 runs, the warning signals the curve proposed, with rules
  hashed before any r9 data was read (`PREREG.txt`, `PREREG_STAGE2.txt` and
  their `.sha256` files); only the policy probe's `tail_3` offset was
  testable on r9, and it fired in all 5 runs by 2.06M, the cleanest negative
  included: the test fails (`wr9_r9.txt`). WM measures, at seven checkpoints
  near the rail, the smallest constant nudge to the `tail_3` command's bias
  at which at least 11 of 20 seeds park it at the rail (`PREREG.txt`, hashed at
  02:50:47 UTC; `NOTES.txt`, `wm_analyze.txt`, `wm_posthoc.txt`). VWF
  re-derives both with its own code (`vwf_r9.txt`, `vwf_procx.txt`,
  `vwf_margin_analyze.txt`, `vwf_margin_heldout.txt`, `vwf_bankz.txt`).
  WR9 and VWF have no notes file; the run log's entry timed 04:40 UTC (line
  773, the copy's 14) holds their verified findings.

What each folder keeps (the rule of `../com_revisit/` and `../tail_probes/`):
the files at the folder's top level that are text, its notes, tables,
scripts, logs and non-empty error logs, its run markers, its JSON outputs of
at most 130 KB and its JSONL outputs of at most 60 KB. Not kept: every
subdirectory (EB's `eb_out/`, `eb_out2/` and `eb_old/`, per-rollout JSON and
npz outputs, 643 files, about 248 MB; `eb_tmp/`, a test copy of the update
and its output; `eb_logs/`, the runs' logs, mostly per-rollout JSON lines;
VEB's `ckpt/`, the two checkpoint pairs it fetched, `vb_out/`, per-rollout
JSON, and `logs/`, its runs' logs; EW's and VEW's `states/`, stored states as
npz), the npz file `warning_r9/wr9_procx.npz`, the 31 empty error logs, and
the larger outputs whose `.txt` rendering is kept or that are per checkpoint
or per episode: `early_bias/eb_aggregate.json` (1,737,517 bytes) and
`eb_actual.json` (500,107), `verify_early_bias/vb_aggregate.json`
(198,020), `early_warning/ew_events.json` (5,927,309), `ew_ecs.jsonl`
(1,695,655) and `ew_r10.jsonl` (376,598),
`verify_early_warning/vew_rows.jsonl` (178,398),
`warning_margin/episodes.jsonl` (700,549) and
`verify_warning_followups/vwf_margin_episodes.jsonl` (214,323). The durable
branch `claude/handoff-bias-screen` holds every top-level file of the seven
folders but `warning_r9/wr9_procx.npz` and `early_warning/ew_events.json`
(over 5 MB); those two and the subfolders stay in the session's scratch only
(that branch's `SKIPPED.txt` lists the subfolders and `ew_events.json`). The
checkpoints the probes read are the first r10 round's local copies and the
entropy screen's continuations, in the session's scratch (the round's are
also on Drive); none is here.

The scripts read the session's scratch paths they name and are kept as
records of what was run, not to be run from here; none reads a name
`test_policy_loading.py` guards.

Kept as their sources have them, so to leave out of a manual pre-commit run:
`early_bias/eb_proxy.json`, `eb_realized.json` and `eb_realized_s60early.json`,
`verify_early_bias/vb_realized.json`, `verify_early_warning/vew_events.json`,
`warning_margin/wm_predict.json`, `wm_static.json` and `wm_summary.json`, and
`warning_r9/wr9_r10_rows.json`, `wr9_r10_unnamed_rows.json` and
`wr9_r9_rows.json` have no final newline; `early_bias/eb_summary.txt` and
`warning_margin/wm_analyze.txt` end in a blank line;
`early_bias/eb_aggregate.txt` (21 lines), `eb_summary.txt` (1),
`early_warning/ew_analyze.txt` (27), `verify_early_bias/vb_aggregate.txt`
(1), `warning_r9/wr9_r10_repro.txt` (21), `wr9_r10_repro2.txt` (21),
`wr9_r10_unnamed.txt` (10) and `wr9_r9.txt` (8) have lines ending in spaces.
