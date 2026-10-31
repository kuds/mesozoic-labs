# The why-probe: why the critic pays for a raised tail (2026-10-30)

Evidence for decision D-D59
([BEHAVIOR_RECIPES_PLAN.md](../../../BEHAVIOR_RECIPES_PLAN.md) §6.2) and for
§10 of [TREX_TAIL_SATURATION_HABIT_2026_10.md](../../TREX_TAIL_SATURATION_HABIT_2026_10.md),
where its short names are given (PH, TP and VW). The maintainer asked for it
at 16:42 UTC on 2026-10-30, after asking why blinding the critic would be a
good fix ("Probe it now, no training (Recommended)"): two lenses, each
pre-registering its predictions with a sha256 before any data, and an
independent verifier that also adjudicated between them. Simulation, network
evaluation and offline regressor fits only, on physics r10 with the sealed
code copy of the never-merged run branch at `8737cd3d` (code tree digest
`331fe8fc…`), at seed 42's r10 checkpoints at 1,048,576 and 2,097,152 steps,
at the second critic-blind screen's continuations at 3,145,728 and 3,538,944
(two controls and one blinded run) and at seed 80's 4,194,304 checkpoint;
no policy was trained or updated. Exploratory and report only. Nothing here
changes a gate, reward, plant or criterion, and nothing here is a gate
reading. The run log of that period is in
[`../phase_critic_results/`](../phase_critic_results/README.md) (its lines 2
and 3, the source's 803 and 804).

[SOURCES.txt](../SOURCES.txt) lists each file's source, its sha256 and the
changes the copy made; [SHA256SUMS](../SHA256SUMS) pins them (every file of
the folder but this README). Each folder here is the folder of the same name
under `scratchpad2/com_revisit/notes/`. Every copy is byte-identical to its
source, absolute session-scratch paths included, but
`verify_why/vw_redo.py`, kept gzipped as `verify_why/vw_redo.py.gz`
(**gzip**: its line 260 has an f-string whose replacement field indexes a
dictionary with a key in the same double quotes, which Python 3.12 allows
and Python 3.11 cannot parse, and
`test_policy_loading.py` parses every Python file under `docs/` on each
Python version CI runs; gunzip restores it byte for byte); none held a
session tool id. Every source's sha256 equals its entry in the SHA256SUMS of
the end-of-day safety copy, the orphan branch `claude/handoff-day-2026-10-30`
at `b25eeaf5` (its `why_probe/`, read in its local stage), never merged.

- **`phase_probe/`** (PH): lens A, the phase proxy. `PREREG.txt` (sha256
  `639cd879…`, its `PREREG.sha256`, recorded at 16:56:47 UTC before any
  rollout) states predictions A1 to A5; `phase_rollouts.py` records 48
  training-like episodes (sampled actions, frozen normaliser) at each of 7
  checkpoints (P1 and P2: s42 at 1,048,576 and 2,097,152; C11a, C11E, C31a
  and C31E: the second critic-blind screen's controls 2511 and 2531 at
  3,145,728 and at E, 3,538,944; B11E: its blinded run 2511 at E) and 32 at
  s80's 4,194,304 (N80); `phase_analyze.py` is the pre-registered analysis,
  `analysis.txt` its output and `verdicts.txt` the verdicts (A1 to A5 and
  A4c SUPPORTED, overall A SUPPORTED); `NOTES.txt` holds the summary, the
  post hoc readings (`posthoc_push.*`, `posthoc_joints.*`, `posthoc_other.*`,
  `posthoc_bs2_probe.*`) and the log; `analysis_test*` and `verdicts_test*`
  are the test runs on two labels each, before the full run.
- **`termination_probe/`** (TP): lens B, the termination risk (does the
  critic value a raised tail because a lower tail risks the tail-floor
  contact that ends an episode?). `PREREG.txt` (sha256 `7c69819d…`,
  recorded at 17:00:42 UTC) states its items; `probe_static.*` is the static
  kinematics, `run_probe.py` and `tp_common.py` the rollouts (128 episodes per
  condition), `analyze_records.py` and `analyze_rollouts.py` the analyses of
  the training records and the rollouts, `posthoc.*` and the `debug_*` files
  its post hoc reading and debug passes, `check_blind_equiv.*` and
  `check_minima.*` two checks; `NOTES.txt` holds the summary (B-genuine NOT
  SUPPORTED, B-apparent NOT SUPPORTED by its own rules) and the log.
- **`verify_why/`** (VW): the independent verifier and adjudicator. It
  checked both pre-registrations' hashes and times, re-derived both lenses'
  numbers with its own code (`vw_redo.py.gz`, `vw_redo_term.py`:
  `redo_phase.*`, `redo_term.*`), recorded fresh rollouts on new seeds
  (`vw_roll.py`, `vw_fresh.py`, `run_fr.sh`: `fresh.*`), pre-registered its
  own checks (`PREREG.txt`, sha256 `8267fcab…`), and read post hoc
  (`vw_posthoc.py`, `vw_ph5.py`: `posthoc.*`, `posthoc_ph5.*`). Its
  `NOTES.txt` ends with the summary and the adjudication: A, the phase
  proxy, favoured strongly; B-genuine excluded; B-apparent unresolved, at
  most a small modulator within a phase; its prediction K1b failed, so it
  reads the termination probe's M3.1 as phase-confounded.

What each folder keeps (the rule of `../com_revisit/`, `../tail_probes/` and
`../overnight_probes/`): the files at the folder's top level that are text,
its notes, tables, scripts, logs and non-empty error logs, its run markers
and its JSON outputs (none here is over 130 KB). Not kept: every
subdirectory (PH's `rollouts/`, the per-label rollouts as npz with their
JSON metadata, and `logs/` and `smoke/`; TP's `runs/`, the per-condition
rollouts as npz with their JSON metadata and episode lines, and `tmp/`;
VW's `runs/`, its fresh rollouts, and `logs/`), and the 11 empty error and
output logs. The end-of-day safety copy holds every file of the three
folders but the npz rollouts; those stay in the session's scratch only. The
checkpoints the probes read are the first r10 round's local copies and the
second critic-blind screen's continuations, in the session's scratch; none
is here.

The scripts read the session's scratch paths they name and are kept as
records of what was run, not to be run from here; none reads a name
`test_policy_loading.py` guards.

Kept as their sources have them, so to leave out of a manual pre-commit run:
`phase_probe/analysis.json`, `analysis_test.json`, `analysis_test2.json`,
`posthoc_bs2_probe.json`, `posthoc_joints.json`, `posthoc_other.json` and
`posthoc_push.json`, and `verify_why/fresh.json`, `posthoc.json`,
`posthoc_ph5.json`, `redo_phase.json` and `redo_term.json` have no final
newline. No copy holds a model identifier, an email address or a commit
trailer.
