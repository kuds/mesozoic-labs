# The tail clamp test, the normaliser probe and the posture tests (2026-10-29)

Evidence for decisions D-D53 to D-D55
([BEHAVIOR_RECIPES_PLAN.md](../../../BEHAVIOR_RECIPES_PLAN.md) §6.2) and for
`docs/investigations/TREX_TAIL_SATURATION_HABIT_2026_10.md` (§3.6, §4.7, §4.8
and §4.11), where each folder's short name is given below. Three no-training
experiments on the T. rex tail saturation habit, each run by one agent and
checked by an adversarial verifier with its own code, on the 24 end policies
of the entropy continuation screen
([`../ecs_results/`](../ecs_results/README.md)), their two start checkpoints
(`stance6_r10LTSB_s42` at 3,145,728 steps and `_s80` at 4,194,304) and, for
the posture tests, the first r10 round's local checkpoints and records.
Simulation, statics and network evaluation only, on physics r10 (`main` at
`378ae2f8`); exploratory and report only, not pre-registered. Nothing here
changes a gate, reward, plant or criterion, and nothing here is a gate
reading. The centre-of-mass revisit they followed is
[`../com_revisit/`](../com_revisit/README.md).

[SOURCES.txt](../SOURCES.txt) lists each file's source, its sha256 and the
changes the copy made; [SHA256SUMS](../SHA256SUMS) pins them (every file of
the folder but this README). The sources are in the session's scratch: each
folder here is the folder of the same name under
`scratchpad2/com_revisit/notes/`, but `normaliser/` (its `probe/`) and
`verify_normaliser/` (its `verify_probe/`). Every copy is byte-identical to
its source, absolute session-scratch paths and checkout paths included;
none held a session tool id.

- **`clamp/`** (CLAMP) and **`verify_clamp/`** (VCLAMP): the tail clamp test
  of D-D53, asked for at 12:42 and 13:10 UTC. The 12 s42 end policies, all
  railed, with `tail_3_pitch_act`'s command held at 0, and the 12 s80 end
  policies with it forced to +1, each against the same policies unforced,
  deterministic, seeds 3042-3061, 1,000 steps (2,000 on 3 per arm); an
  open-loop replay of each episode's own commands, the statue, the two
  starts, and D-9's hold reproduced first. The verified readings: command +1
  holds the tail tip at about +5.2°, not the 12° stop; on the plant alone the
  rail moves the centre of mass 0.2 mm forward and 1.6 mm up; holding it at 0
  gains 63.6 reward per episode (12 of 12) with the stance otherwise
  unchanged; forcing it on s80 costs 180 (12 of 12); and the verifier's
  observation-only sweep (`verify_loop.py`, `loop_summary.txt`) measures the
  positive feedback from the observed tail angle to the tail command, loop
  gain about 0.63 at the s42 ends against 0.27 at their start. `NOTES.txt` in
  each; `clamp/analysis.txt`, `rail_onset.txt`, `d9_repro.txt` and
  `probe_signs.txt` and `verify_clamp/static_tail.txt` and `onset.txt` are
  the tables the record cites.
- **`normaliser/`** (PROBE) and **`verify_normaliser/`** (VPROBE): the
  normaliser probe of D-D54 (4). All 168 observation-normalisation pickles
  of the 24 continuations read, the loop gain swept at 146 checkpoints, the
  gain split into a normaliser part and a weight part on fixed states, and
  2,680 closed-loop episodes with swapped statistics. The answer is no: at
  fixed weights the drift of the tail_3 angle's statistics lowers the gain
  (the end weights with the start's statistics 1.143 against 0.630, 12 of
  12), and the growth sits in the weights downstream of the first layer. The
  verifier reproduces it bit for bit and corrects how much of it each part
  carries (by sweep range); whether the drift shaped training is untested.
  `stats.txt`, `static.txt`, `swaps.txt`, `weights.txt`, `hybrid.txt`,
  `mean_vs_var.txt` and `implied_window.txt` are the probe's tables, with
  the scripts and `run_step2.sh`/`run_step3.sh` that made them;
  `verify_normaliser/v_*.txt` are the verifier's.
- **`posture_timing/`** (PTIME), **`posture_policy/`** (PPOL) and
  **`verify_posture/`** (VPOST): the three posture tests of D-D55 (2). Test 1
  (`posture_timing/`, records only, no simulation): whether a posture shift
  came before the tail rose, across the r9 and r10 run records and the 24
  continuations. Tests 2 and 3 (`posture_policy/`): whether the policies'
  tail command responds to posture inputs (sweeps, transplants, a
  closed-loop pitch offset), and whether the plant pushes the tail up, or
  makes tail-up cheaper, in the railed lineage's posture. The verifier
  re-derives test 1 with change points and a block bootstrap, re-measures
  tests 2 and 3 on its own states, and probes the r10 checkpoints at
  1,048,576 and 2,097,152 and s60. The verified readings: no mechanical push;
  no nose-down or forward centre-of-mass shift before the tail rose; in the
  s42 lineage the tail bias rose first (about 1.20M steps), and from mid-rise
  on its networks read their nose-down posture as a tail offset that the loop
  amplifies, a pathway absent at 1,048,576. `NOTES.txt` in each;
  `posture_timing/r10_timeline.txt`, `prop_timing.txt`, `ecs_comovement.txt`,
  `ts_comovement.txt` and `bank_drive.txt`, `posture_policy/sweep.txt`,
  `sweep_extra.txt`, `analysis_cl.txt`, `analysis_long.txt`,
  `analysis_plant.txt`, `plant_pitch.txt` and `cross_policy.txt`, and
  `verify_posture/vt1_timing.txt`, `vt1b_onset.txt`, `vt1c_boot_ecs.txt`,
  `vp_prerise.txt`, `vp_s60.txt`, `vp_sweep2.txt` and `vp_analyze.txt` are
  the tables the record cites.

What each folder keeps (the same rule as `../com_revisit/`): the files at the
folder's top level that are text, its notes, tables, scripts, logs and
non-empty error logs, its run markers, its JSON outputs of at most 130 KB and
its JSONL outputs of at most 60 KB, and from a verifier's `out/` folder its
text outputs, logs and run markers. Not kept: every other subdirectory (the
clamp's per-episode rollouts in `run/`, `long/`, `replay/`, `start/` and
`smoke/`; the probe's `ckpts/` and `swaps/` episodes and stored states; the
posture policy test's `run/` and `test/`; the verifiers' code copies and
per-episode JSONL), the `.npz` files (among them `verify_posture/`'s 26
settled-state files, about 0.57 MB each), and the larger outputs whose `.txt`
rendering is kept (`clamp/analysis.json`, `normaliser/stats.json`,
`verify_normaliser/v_stats.json`, `posture_policy/sweep.json`) or that are
per-episode or per-state (`posture_policy/plant_static.json` and
`plant_replay.jsonl`, `verify_normaliser/v_swaps.jsonl`). The checkpoints
they read are in the session's scratch (the screen's, and the first r10
round's local copies, which are also on Drive); none is here.

Kept as their sources have them, so to leave out of a manual pre-commit run:
`normaliser/layout.json`, `static.json` and `swaps_summary.json`,
`posture_timing/ecs_series.json` and `r10_timing.json`,
`posture_policy/plant_pitch.json` and `sweep_extra.json`, and
`verify_posture/vp_sweep2.json` and `vt1_timing.json` have no final newline;
`posture_timing/bank_drive.txt`, `prop_timing.txt` and `r10_timeline.txt`,
`verify_posture/vt1_timing.txt` and `vt1_timing_lib.py` end in a blank line;
`posture_timing/r10_timeline.txt` has 126 lines and
`posture_policy/cross_policy.txt` 2 lines ending in spaces.
