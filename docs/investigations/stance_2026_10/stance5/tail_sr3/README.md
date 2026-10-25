# T. rex physics r10, the preloaded tail springs (decided and built 2026-10-25)

Evidence for §25 of
[TREX_STANCE_SYMMETRY_AND_FOOT_2026_10.md](../../../TREX_STANCE_SYMMETRY_AND_FOOT_2026_10.md):
the posture options study that followed the tail_3 bound screen (§24), the
r10 tail probe the maintainer then asked for, the CPU checks of its candidate
SR3, and the re-measurement of every constant pinned to physics r9 on the
branch that builds it, decision D-D44
([BEHAVIOR_RECIPES_PLAN.md](../../../../BEHAVIOR_RECIPES_PLAN.md) §6.2). SR3
gives `tail_1_pitch`, `tail_2_pitch` and `tail_3_pitch` a spring rest angle
(`springref`) of 52.5, 25.9 and 10.4 degrees and changes nothing else: the
compiled model differs from r9's only in `qpos_spring`. Nothing here trained a
policy or took an optimizer step; the probe and the CPU checks ran on scratch
MJCF copies and scratch trees, and the r10 branch (`claude/physics-r10-sr3`,
a scratch clone of `main` at `3ffc18e9`, push disabled) was built and measured
on this container's CPUs. "Physics r10" here is the plant counter's next
integer on `main`, not the foot that the parked draft #623 screened under that
name (D-D34; its evidence is [`r10_screen/`](../../r10_screen/)), which never
became part of `main`.

These are copies taken on 2026-10-25 from the second session's scratch tree
(`scratchpad2/tailround/posture/`, `r10probe/` and `r10cpu/`, and
`scratchpad2/prs/r10_work/build/`), and the session's run log from the first.
[SOURCES.txt](../../SOURCES.txt) lists each file's source, its sha256 and the
changes the copy made; [SHA256SUMS](../../SHA256SUMS) pins every file but this
README.

**Outcome.** At 04:55 UTC on 2026-10-25 the maintainer approved (1) the r10
revision as SR3, physics revision 9 → 10 with policy interface r15 kept, no
training, every r9-pinned constant and golden re-measured, and no pull request
until asked, the first r10 training round to get its own criteria; (2) the
reward change that re-centres `tail_home_pose`'s target on SR3's measured
settled pose, its tolerance kept at 0.05 rad; and (3) the heading-invariance
test's fix as its own test-only pull request (#642, opened at 07:35 UTC). The
re-measured constants are in `remeasure/`; the branch carries them.

## The files

Measured and inferred are labelled as in their sources. Where a later check
corrected a file, the correction is given here and in §25; the file is kept as
it was written.

- **`results/run_log.txt`**: lines 578 to 600 of the session's run log (the
  entries timed 01:01 to 08:21 UTC on 2026-10-25; first copied to line 598 and
  extended in the branch's review by the heading-test fix's two entries, timed
  07:35 and 08:21 UTC), workflow and scheduled-task ids removed; lines 505 to 577 are
  [`../tailscreen/results/run_log.txt`](../tailscreen/results/run_log.txt).
  The decisions are its entries timed 01:33, 02:18 and 04:55 UTC.
- **`posture/`**: the posture options study (01:02–01:31 UTC; no training).
  `brief_draft.md` is the synthesis brief before its two skeptics;
  `brief_corrected_summary.md` is the corrected summary that governs it. The
  skeptics found four blocking problems in the draft's framing: the PARKED rule
  exists so that a bound that only masks the outward drive is not credited, not
  as an anatomy rule; anatomy does not rank the r9 postures (they differ by 1.5
  to 3 degrees inside a band 22 to 28 degrees below the level tail that
  reconstructions softly favour); a servo redesign sized just to hold a level
  tail (the break-even kp and control range) holds it only at a command of +1,
  the rail, and r9's own servos cannot reach it at all, so passive support
  (`springref`) is the route off it;
  and two CPU options were missing. Measured there: under the 0.60 bound the
  tail_3 joint sits at −2.28 degrees, 1.2 degrees lower than the walking-pilot
  parent's; in every arm the distal tail points 22 to 28 degrees below
  horizontal and `tail_1` sits on its ventral stop; 52 to 59% of the training
  rows still had `tail_3`'s mean above 0.60.
- **`probe/`**: the r10 tail probe (01:34–02:07 UTC; CPU only, scratch MJCF
  copies). `brief_draft.md` is the probe's brief before its skeptic;
  `chainsizer/results_v1.txt` the candidates settled at zero action and swept
  for reach; `chainsizer/panels/` the statue panels' summaries (r9 `C0_r9`,
  SR3 with the tail target re-centred `SR3_recentred` and with r9's
  `SR3_r9target`, SR2 and stiffer servos, seeds 7042–7081, and the open-loop
  command-noise panels at SD 0.10 with their fall directions); `cost/` the
  repository-cost probe (which edits move which digest, the settle on a
  placeholder tree, its targeted tests); `skeptic/` the skeptic's checks. The
  skeptic's findings (2 blocking, 5 minor) were applied in the report to the
  maintainer and are not on disk as a document; the run log's entry timed
  02:07 UTC gives the corrected reading. Two corrections a reader of
  `brief_draft.md` needs: the statue's 40/40 on SR3 holds through the gate's
  own code (`skeptic/g_sr3rc.log`: `StanceV2Result(passed=True …)`, 40 of 40
  clean, reward 3768.85 ± 16.11), where the brief had used its own copy of the
  thresholds; and the heading-invariance test does fail on an SR3 tree
  (`skeptic/head_om_sr3rc.log`: rewards 1.34e-8 against 1e-9), where the brief
  called that untested. Its line "joints differed by 7.2e-8 against a 1e-9
  bound" names the wrong bound: the joint bound is 1e-6 and the reward bound,
  1e-9, is what trips (the CPU checks' heading worker,
  `cpucheck/coord/heading_report_text.txt`).
- **`cpucheck/`**: the CPU checks (02:36–04:27 UTC; four agents, no training).
  `coord/` holds, per worker (zero-shot, gate, heading, skeptic), its summary,
  report, findings, problems and what it dropped, as the coordinator wrote
  them out of the checks' result file. `zeroshot/` holds the zero-shot closed
  loop of eight r9 checkpoints on r9 and on SR3 (`checkpoints.csv` with each
  pair's sha256; `validation.json`, the harness reproducing recorded r9
  panels bit for bit; `summary.csv`, `paired_tests.csv` and `pooled.json`;
  `head_neck.csv`). `gate/` holds the gate worker's SR3 tree as a patch on
  `9f2355a3` (`sr3.patch`), its settle (`settle_sr3.json`), the 280-seed and
  six-heading statue panels' summary (`panels_summary.json`), the statue
  reference values (`statue_reference.json`), the 13 tests that fail on SR3
  only and what each pins (`test_classification.json`), the pad-CoP fixture's
  lean sweep (`cop_fixture_diagnosis.json`) and the refusal of an r9 checkpoint
  on SR3 (`checkpoint_refusal.json`). `heading/` holds the heading-invariance
  diagnosis: the test's own check over seeds 3042–3061 on both trees
  (`q1_summary.json`) and the step-by-step analyses of where a pair first
  differs (`q2_*_analysis.json`, `q2_funnel.json`). `skeptic/` holds the
  skeptic's per-claim verdicts (`verdicts.csv`), its statistics
  (`power_and_stats.json`), its recount of the zero-shot panels
  (`zs_recount.json`) and its re-run of seven statue seeds against the gate's
  (`statue_rerun_vs_gate.csv`).
- **`remeasure/`**: the re-measurement on the r10 branch (05:34–06:21 UTC; the
  build's logs). `remeasure_summary.json` lists each constant, its old and new
  value, the procedure and the file; `settle_r10_final.json` (the tail and neck
  targets, `natural_pitch`, `target_z`, the head tip, the 40-seed spread),
  `legs_r10_final.json` (`_LEG_SETTLED_QPOS`, steps 200–999), `mgh_r10.json`,
  the statue references (`zab_stage1_r10.log` to `zab_stage3_r10.log`,
  `sqb_stance_r10.log`, `recovery_statue_r10.{json,log}` and
  `recovery_statue_ends_r10.json`), the CoP golden and the fixture's sweep
  (`cop_r10.json`, `cop_fixture_sweep_r10.json`), the moved digest lines
  (`digest_moved_lines.txt`), the compiled-model comparison
  (`env_model_diff.json`) and the noise-free pad tilt
  (`sole_tilt_noise_free.txt`). Each `*_main_r9.*` file is the same procedure on
  a `git archive` of `3ffc18e9` (physics r9), run first. It gives r9's own
  figures: they equal the committed values where r9 set them (`forward_z`, the
  settled leg pose, the CoP golden) and differ slightly where the committed pin
  is older than r9 and r9 carried it (the tail and neck targets, within 1.6e-4
  rad; the m·g·h literal 704.4, r8's, against 704.57; the recovery reference
  1329.1 against 1369.4, 0.5 standard errors). No r9 log of the three
  zero-action references is here; their r9 figures are the gate worker's on
  `9f2355a3`, the same code (`cpucheck/gate/statue_reference.json`). `panels/`
  holds the r10 statue panels' summary (the 280 seeds at 0 degrees and the
  six-heading certificate on 3042–3081), the certificate's text report, and
  `stance_heading_report_final_gate.txt`, the certificate re-rolled in the
  branch's review under the final stance gate block (gate `d26c34e6…`, the
  rail 2260): 40/40 at every heading, as before. `scripts/` holds the build's measuring scripts, the Python ones
  gzipped (gunzip restores them; they import the repository's test helpers and
  take the tree as their first argument).

## Corrections and limits that apply to the files

From the CPU checks' skeptic (`cpucheck/coord/skeptic_*.txt`; no blocking
finding, 14 minor):

- The zero-shot pooled McNemar p-values (9.0e-10 deterministic, 3.7e-9
  stochastic) treat 320 pairs as independent and overreach: 80 of the 88
  deterministic losses come from two checkpoints (c3 and c5), and at the
  checkpoint level 6 go down and 2 up (sign test p 0.29; stochastic 7 and 1,
  p 0.07).
- Zero-shot gives no sign that SR3 cures the rail: actuator-saturation failures
  rise from 27 to 80 (deterministic) and 74 to 130 (stochastic), tail-up
  episodes from 27 to 40 and 73 to 90; c4 losing its rail is offset by c5
  gaining one. The tail observations sit 5 to 8 SD outside the r9 normalizer's
  statistics, so zero-shot cannot predict a policy trained on SR3.
- The zero-shot worker's "balance is not the limiting factor" is overclaimed:
  0 falls in 40 episodes is compatible with a 2% fall rate (P 0.45), the SR3
  statue falls backward on 1 of 280 seeds (9047) and the probe's open-loop
  falls shifted backward; backward-fall risk under perturbation is untested at
  useful power.
- The zero-shot reward numbers (return deltas −139 to −578, `tail_home_pose`
  −122 to −171, the statue 4456.3 → 4285.3) come from an SR3 env that kept r9's
  tail target, so they are not r10 values; gate classification does not read
  the reward.
- Two zero-shot panel verdicts on SR3 are one episode from flipping: c1 passes
  at 37/40 (bound 0.817) and c4 fails at 36/40 (0.786).
- The walking-pilot parent (c1) and c2 drop their heads on SR3 (head tip 1.24
  → 0.61 m and 1.08 → 0.67 m) and c1 still passes the stance gate: a blind spot
  of the gate, not a property of SR3.
- The gate worker's "not a loss of gate margin" for the pad-CoP fixture is
  inferred: the lean window where only the CoP bar trips is one step wide on
  both plants (corner-lift headroom 0.21 mm on r9 at −0.0825, 0.26 mm on SR3 at
  −0.09). The gate worker did not re-measure the recovery stage's statue
  reference (`remeasure/` does).
- The heading report's "l_hip_pitch:qvel (index 38)" is observation index 35
  (38 is `l_ankle:qvel`); the analysis JSON records 35. Fisher p 1.0 between
  the trees' failing seeds (7 and 6 of 20) shows no evidence of a difference,
  not equal fragility. The heading worker's SR3 tree used an earlier tail target
  (0.00013, 0, 0.00014, 0.00023); its rewards move by about 2e-13 and no
  verdict changes.
- None of the checks tests whether a policy trained on SR3 avoids the tail_3
  rail, the core of the question.

From the build: the six-heading certificate and the 280-seed panels in
`remeasure/panels/` were rolled under the final stance task (digest
`533fff20…`) and the stance gate block as it stood before its rail moved from
2250 to 2260 (gate digest `06375c21…`, r9's). The branch's review re-rolled
the certificate under the final block (gate `d26c34e6…`;
`remeasure/panels/stance_heading_report_final_gate.txt`): 40/40 at every
heading, as before (measured). The 280-seed panels were not re-rolled; the
rail is a panel rail that no episode's classification reads, and the statue's
3766.6 clears it, so their verdicts hold under the final block (inferred). `settle_r10.json`, an earlier
settle taken before the env defaults were edited, is superseded by
`settle_r10_final.json` and not copied; every dynamics-only figure in the two is
identical.
