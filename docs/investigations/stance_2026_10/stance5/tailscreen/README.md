# T. rex tail_3 bound screen (run 2026-10-24, scored and verified 2026-10-25)

Evidence for §24 of
[TREX_STANCE_SYMMETRY_AND_FOOT_2026_10.md](../../../TREX_STANCE_SYMMETRY_AND_FOOT_2026_10.md):
the round on the T. rex tail saturation habit that decision D-D42 (1) asked for,
and the screen the maintainer decided on as decision D-D43
([BEHAVIOR_RECIPES_PLAN.md](../../../../BEHAVIOR_RECIPES_PLAN.md) §6.2). It is a
scratch CPU continuation screen of a training-loss bound on the policy's
Gaussian mean for `tail_3_pitch_act` (actuator index 14), upper side only, in
three arms: CONTROL (coef 0), BOUND90 (b = 0.90) and BOUND60 (b = 0.60), coef
2.4 for both bounds. Each of 18 replicates (3 arms, 2 start states, 3 seeds)
continued one 1,048,576-step run from a band checkpoint at 4,194,304 steps:
stance5 s42's (on the rail, "persistence") or stance5r s80's (clean, the run
took the habit later, "onset"). The bound's code is two scratch commits on the
band branch's `cf80ecb1`, never pushed and never merged; the screen ran on this
container's CPUs, not on Colab. A screen certifies nothing: the verdicts are the
fixed criteria's, nothing here certifies a run, writes a result bundle or
promotes anything, and nothing the screen ran is on `main`; `main` and its
digests are unchanged.

These are copies of the files that fixed, gated, scored and verified the screen,
taken from the second session's scratch tree (`scratchpad2/tailround/screen/`)
on 2026-10-25. The options study and the no-training probes that came before it
are in [`options/`](options/) and [`probes/`](probes/) (the folder README's
entries of 2026-10-25). Not here: the 18 run folders (all nine checkpoint pairs
per replicate, `run_meta.json`, `progress.csv`, `telemetry.npz`), the 162
panels' per-step CSVs and the start checkpoints. The run folders' R1 and final
checkpoint pairs, `run_meta.json`, `progress.csv` and `telemetry.npz`, every
panel's `episode_trace.json`, `flags.csv`, `csv_sha256.txt` and
`trace_run.json`, the start pairs, the patch and the tools are on the orphan
branch `claude/handoff-tailscreen` (head `1106cd3f`, never merged, never a pull
request; the maintainer's decision (3) of 17:46 UTC); the other periodic
checkpoints and the per-step CSVs (gzipped) are in the session's scratch only.

**Outcome: C, final and verified. BOUND90 NOT ADOPTED (RAIL); BOUND60 NOT
ADOPTED (PARKED).** In the criteria's words, "the lower bound cleared the rail
but not the posture". BOUND90 goes to the maintainer's choice of a next
candidate (the deterministic-state variant, a stronger dose, or the posture
decision); BOUND60 goes to the posture decision (ruling (3) of 16:00 UTC). The
outcome changes no screen, verdict, parent, criterion or stage config, and
nothing trains on it without the maintainer's approval. Fails stay fails: no
replicate was stopped, changed, extended or rerun on a readout.

| State | Arm | r101 | r202 | r303 | R1/R2 mean CMD (r101, r202, r303) | Tail-up flags of 40 | State reading |
|---|---|---|---|---|---|---|---|
| s42 | CONTROL | RAIL | CURED | RAIL | 0.9856, 0.2652, 0.9023 | 36, 0, 38 | (shared) |
| s42 | BOUND90 | RAIL | RAIL | RAIL | 0.8864, 0.9236, 0.9507 | 1, 14, 34 | RAIL |
| s42 | BOUND60 | PARKED | PARKED | PARKED | 0.6070, 0.6002, 0.6128 | 0, 0, 0 | PARKED |
| s80 | CONTROL | RAIL | RAIL | RAIL | 0.9371, 0.9106, 0.8444 | 23, 24, 13 | (shared) |
| s80 | BOUND90 | RAIL | RAIL | RAIL | 0.9003, 0.9205, 0.9708 | 4, 18, 32 | RAIL |
| s80 | BOUND60 | PARKED | PARKED | PARKED | 0.6318, 0.6280, 0.6152 | 0, 0, 0 | PARKED |

Read at R1 = 5,111,808 and R2 = the final, 5,242,880 (20 deterministic episodes
each, seeds 3042–3061, window steps 200–999). CMD is the window-mean `tail_3`
command; a TAIL-UP FLAG is an episode whose share of window steps at a command
of +0.99 or more exceeds 0.10. RAIL: at least one tail-up flag in the 40
episodes of R1 and R2; PARKED: no flag and the R1/R2 mean CMD at least 0.60 (no
upper end, the literal ruling the maintainer kept at 17:46 UTC); CURED: neither.
OTHER flags (another actuator at saturation, `tail_3` at −0.99, or a short
episode) are 0 in every arm at R1 and R2, so no bound reads OTHER, and NOT
SEPARATED does not apply (1 of 3 s42 CONTROL replicates CURED; the line needs
2). The onset note is INFORMATIVE: all three s80 CONTROL replicates had tail-up
flags, each with its 9 checkpoints measured. Report only, never in a verdict:
the dose ordering BOUND60 ≤ BOUND90 ≤ CONTROL of the arms' mean CMD is false at
both states (s42 0.6067, 0.9203, 0.7177; s80 0.6250, 0.9305, 0.8974), because
BOUND90 sits above CONTROL.

The verification (00:13 UTC on 2026-10-25, three independent checks and a
synthesis, no blocking finding) re-scored every class, state reading, the onset
note, both verdicts and the outcome from `criteria.txt` alone on the 720 R1/R2
episodes' per-step traces; checked the provenance (18 of 18 replicates done with
the pinned arm, coef, b, seed and state; every panel row's checkpoint sha256
equal to its file; the pins, the scratch clone at `4c0c8da7`, clean, and the
judge clones clean; the durable branch's `score.json` byte-equal); re-measured
four final panels field for field (`flags.csv` byte-identical); and checked that
the bound acted (the manipulation check, report only): CONTROL's applied bound
loss is exactly 0 in all 128 updates of its six runs, both bound arms log coef
2.4 and a non-zero loss in 128 of 128 updates, and in every seed-paired
comparison the share of training rows with `tail_3`'s mean above b is lower
under the bound (at b 0.90, s42 0.611, 0.444 and 0.503 against 0.249, 0.330 and
0.237; s80 0.391, 0.316 and 0.332 against 0.194, 0.242 and 0.243). Its minor
notes, none changing the outcome: all six BOUND60 means lie within 0.032 of
0.60, s42_b60 r202 clearing it by 0.00017 and r101 by 0.0070 (the criteria's
KNOWN LIMITS name this), but BOUND60's verdict also holds on s80 alone (3 of 3
PARKED, +0.015 to +0.032 over the line); s42_b90_r101's RAIL rests on one R2
episode at a tail-up share of 0.111; BOUND60's command moves between 0.575 and
0.688 over its checkpoints, not pinned at b; `PINS.txt` is a stale pre-approval
snapshot, though every sha256 in it matches the current files; and `APPROVED`'s
`runlog_line` 551 is the maintainer's re-pin decision, while the criteria's
sha256 first appears on line 552.

Report only, from `results/panels.csv` (measured): one panel of the 162 has a
fall, `s80_b60_r303` at 4,980,736 (the episode of seed 3050 ended at step 504);
it is not a read checkpoint. BOUND60's highest window raw maximum (the raw mean
at any window step) over every checkpoint is 0.858, at `s80_b60_r303`'s
4,849,664, not a read checkpoint; its largest window raw mean is 0.688, at
`s80_b60_r202`'s 4,456,448. At R1 and R2 the arms' mean returns are 4393.3,
4378.3 and 4414.5 at s42 (CONTROL, BOUND90, BOUND60) and 4385.3, 4384.0 and
4405.1 at s80, not independent evidence (the reward contains `action_saturation`
and `tail_home_pose`). The 0.5M readings (4,587,520 and 4,718,592) are in
`results/score.json`'s `report_only.half_point`, shown, never read.

## The files

- `criteria.txt`: the fixed criteria, byte-identical to the text the maintainer
  approved and the gate re-pinned (sha256 `9930ee72…`, 566 lines with its
  appendices; chmod 0444 in the scratch). Why it is run, the predicted outcome
  (written before anything ran: BOUND90 most likely NOT ADOPTED, with s42
  reading RAIL or PARKED and s80 reading PARKED; BOUND60 open),
  what is trained, the coef rule and its calibration, the runs, how they are
  measured, the classes and the decision, run handling, when the decision is
  read, the pre-registered readouts, measured against inferred, known limits,
  compute, the build list, the pins, a reference table and three appendices (V2,
  F, G). Its `SP`, `S2` and `/home/user/...` paths name the two scratch trees
  and the session's checkout. The first approved text (`efd8e297…`) differs only
  by the re-pin that APPENDIX G records; it is not copied.
- `APPROVED`: the approval record as rewritten after the re-pin (`42627d1f…`,
  `runlog_line` 551, `approved_utc` 19:10:16 UTC), byte-identical.
  `APPROVAL_REQUEST.md`: what the maintainer was asked at 19:05 UTC, with the
  two choices it put (the COEF level: 2.4 by the rule as written, 2.7 at P1's
  per-row stiffness; and the reading of "per bound"); the maintainer approved
  2.4 for both bounds. `PINS.txt`: the pin list, kept as it was and stale: its
  header still reads as written at 19:04:12 UTC, before the approval ("NOT
  APPROVED"), though the run log's entry timed 19:10 UTC says it was updated at
  the re-pin, and by the verification every sha256 in it matches the current
  files.
- `calib/coef.json` (`f6d90f99…`, byte-identical; no final newline, kept so its
  sha256 stays the one `APPROVED` and the run log pin) and `calib/calib.log`:
  the calibration of 18:16 UTC, with no optimizer step and the parameter and
  Adam digests equal before and after: the policy-gradient pull on `tail_3`'s
  output bias, RMS 0.513 at s42 and 0.540 at s80, pooled 0.5267 over 512
  mini-batches, so COEF_90 = COEF_60 = 2.4 by the fixed rule (0.5267 / 0.22);
  the realised pull at the start, BOUND90 0.083 at s42 and 0.0015 at s80,
  BOUND60 1.154 and 0.283.
- `patches/tail3-upper-bound.patch`: the two scratch commits, `2cc1921d` (the
  bound and its tests) and `4c0c8da7` (the coef-0 identity tests made
  non-vacuous), as `git format-patch --stdout cf80ecb1..4c0c8da7`; they change
  `environments/shared/mirror_kl_ppo.py` and its test file only. The copy
  carries the **trailer**, **redact** and **whitespace** changes (below), so its
  sha256 is not the `ef0fe36b…` the criteria pin; `git apply` of the copy on
  `cf80ecb1` gives the two files of `4c0c8da7` exactly (checked 2026-10-25).
- `gate/`: the pre-training gate. `first_gate_failed/` holds the first approval
  record (`APPROVED.first`, `runlog_line` 548, criteria `efd8e297…`) and the
  first gate's test outputs: 200 passed, 1 failed, 2 skipped; the failure is
  `test_no_stage_config_enables_symmetry`, `main`'s rule that no shipped stage
  sets `[ppo] symmetry*`, which fails identically on an untouched `cf80ecb1`
  copy because the band arm's `stance.toml` sets it by design (measured). After
  the maintainer's decision of 19:09 UTC the gate deselects that one test;
  `gate_pass.json` (`d7118520…`, byte-identical, no final newline), the passing
  gate's outputs (200 passed, 0 failed, 2 skipped, 1 deselected; head
  `4c0c8da7`) and the two gate logs are the rest.
- `tools/`: the screen's tools as pinned in the run log's entry timed 19:11 UTC
  and in `criteria.txt`'s PINS: `run_tailscreen.py` (one replicate; kept as
  `run_tailscreen.py.gz`, below),
  `calib_tailscreen.py`, `measure_tailscreen.py` (panels, flags, `panels.csv`),
  `score_tailscreen.py`, `screen_common.py` (states, arms, labels, stack, gate
  requirements and every tool's pin), `gate.sh`, `run_waves.sh` and
  `copy_durable.sh`, with the tools' tests in `tools/tests/`.
  `calib_tailscreen.py`, `score_tailscreen.py` and the four tests are
  byte-identical, and `run_tailscreen.py` is once gunzipped: it is kept as
  `run_tailscreen.py.gz` because `test_policy_loading.py`, which parses every
  Python file under `docs/`, would otherwise count it among the repository's
  callers of `_ensure_sb3`; `measure_tailscreen.py` carries **checkout**,
  `screen_common.py` **prefix2** and **prefix**, `gate.sh` and `run_waves.sh`
  **prefix2**, and `copy_durable.sh` **trailer**, **prefix2** and **checkout**,
  so those five do not have their pinned sha256, and the scorer's own pin checks
  fail against these copies.
- `results/`: `score.txt` and `score.json` (written at 23:53:43 UTC, `66c4a92d…`
  and `471ad8e4…`, byte-identical), `panels.csv` (162 rows, one per checkpoint
  measured, 9 per replicate, byte-identical), `run_waves.out` (the waves'
  start and exit times; 18 of 18 replicates exit 0, no crashed attempt) and
  `run_log.txt`, lines 505 to 577 of the session's run log (from the entry
  timed 15:39 UTC on 2026-10-24 to 00:16 UTC on 2026-10-25), following
  `../replication/results/run_log.txt` (lines 396 to 504), as each earlier
  screen's run-log copy sits in its `results/`; 22 workflow run ids (12
  distinct) and one background-job id removed.
- `verify/`: the verification. `REPORT.md` is the synthesis (its first
  paragraph's note on a refused write is about the synthesizing step's own
  attempt; the session wrote the file from that text); `rescore/` the
  independent re-score from the criteria text (`rescore.py`, its output and
  JSON, the comparison with `score.json`, `DIFFS []`, and a threshold
  sensitivity printout; its `meta False` column compares `run_meta.json`'s
  `final_sha256`, a mapping of two files, with one zip's sha256, so it is false
  on every row: the provenance check reads the mapping and finds every final
  equal, and so does a re-read of `s42_b60_r101`'s on 2026-10-25); `provenance/`
  the provenance check and the manipulation check; `remeasure/` the four final
  panels re-measured (`s42_b60_r101`, `s42_control_r202`, `s80_b90_r202`,
  `s80_control_r303`): the comparison, the scripts, and each panel's flags and
  row, without the per-step CSVs.

## What changed in the copies

[`../../SOURCES.txt`](../../SOURCES.txt) lists every file here but this README
with the scratch file it came from (below `scratchpad2`, or `foot/...` for the
run log), that file's sha256, and what the copy changed;
[`../../SHA256SUMS`](../../SHA256SUMS) pins them. The changes are the folder's
**prefix**, **prefix2**, **checkout**, **whitespace**, **trailer**, **redact**
and **lines**, as SOURCES.txt defines them, nothing else:

- **trailer:** the patch's two commits each had two commit-trailer lines, and
  `copy_durable.sh` writes the same two lines into its commits; each pair is
  replaced by `[commit-trailer lines removed]`.
- **redact:** the patch's two `From:` lines (the commits' author name and email
  address) read `From: [name removed]`; in `results/run_log.txt`, the workflow
  run ids (`[id removed]`, or `workflow [id removed]` for a bare one) and one
  background-job id (`[id removed]`).
- **lines:** `results/run_log.txt` is lines 505 to 577 of the session's run log.
- **prefix2:** the second scratch tree's absolute path, replaced by
  `scratchpad2`, in `APPROVAL_REQUEST.md`, `PINS.txt`, `calib/calib.log`, the
  two `gate_junit_tools.xml` (in `gate/` and `gate/first_gate_failed/`), the
  tools `screen_common.py` (which carries **prefix** too), `gate.sh`,
  `run_waves.sh` and `copy_durable.sh`, and the two scripts in
  `verify/remeasure/scripts/`. **checkout** is in `measure_tailscreen.py` and
  `copy_durable.sh`. `criteria.txt` keeps its two absolute session-scratch paths
  (the lines 18 and 19 that define `SP` and `S2`), the only copy here that does,
  because its sha256 is the approved pin. The changes in `options/` and
  `probes/` are in the folder README's note of 2026-10-25.
- **whitespace:** trailing spaces removed (the single space of the patch's blank
  context lines and of its signature separator, which `git apply` still accepts,
  and one line of `PINS.txt`), a final newline added to the four JUnit files and
  to five JSON copies of the verification; `calib/coef.json` and
  `gate/gate_pass.json` keep their missing final newline, so that their sha256
  is the pinned one.

`criteria.txt`, `APPROVED`, `gate/first_gate_failed/APPROVED.first`,
`calib/coef.json`, `gate/gate_pass.json`, the gate's test printouts and logs,
`results/` but `run_log.txt`, `verify/REPORT.md`, the verification's printouts
and `remeasure/compare.json`, and its scripts but the two in
`remeasure/scripts/` (**prefix2**) are byte-identical to their sources. No copy
holds a model identifier, an email address or a commit trailer.

Eight small per-episode files are kept, an exception to the folder's rule that
per-episode files are not copied: the four `final_flags.csv` in
`verify/remeasure/out/` (20 episodes each) and four probe outputs,
`probes/probecheck/probe_s42.json` and `probe_s80.json` (6 episodes each),
`probes/skeptic/noise_shift_s80.json` (90) and
`probes/skeptic/unclip_alt_s42.json` (40), at most 14 KB each, because they are
the evidence of the re-measurement's field-for-field result and of the probes'
counts. Every panel's own `flags.csv` stays on the handoff branch, not here.

## How to re-check

| Check | Command | Result |
|---|---|---|
| The criteria the maintainer approved and the gate re-pinned | `sha256sum criteria.txt` | `9930ee722bf07f21256c0998656568d544ef7413c18d4dfa823338370bcb1b6a`, as in `APPROVED` and the run log's entry timed 19:10 UTC |
| The approval, the calibration and the gate record | `sha256sum APPROVED calib/coef.json gate/gate_pass.json` | `42627d1f…`, `f6d90f99…` and `d7118520…`, the values the run log's entry timed 19:11 UTC records |
| The score | `sha256sum results/score.txt results/score.json` | `66c4a92d…` and `471ad8e4…`, as in the run log's entry timed 23:53 UTC |
| The outcome from the criteria alone (session scratch only: it reads the per-step CSVs and run folders) | `python3 -I verify/rescore/rescore.py SCREEN`, then `python3 -I verify/rescore/compare.py SCREEN`, with `SCREEN` the scratch's `scratchpad2/tailround/screen` | `DIFFS []`, as `verify/rescore/compare.out` |
| The patch on the band branch | in a clone of `claude/screen-trex-r9-armlts-copband` at `cf80ecb1`: `git apply patches/tail3-upper-bound.patch`, then `git diff 4c0c8da7` there (with the scratch commits fetched from the durable branch's patch) | no difference in the two files (checked 2026-10-25) |
