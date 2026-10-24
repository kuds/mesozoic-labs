# T. rex stance after run A: symmetry, the r10 foot and the forward lean

**Date:** 2026-10-17 (the experiments of 2026-10-12 to 2026-10-17)

**Status:** completed screens and analyses, the technical record of the T. rex
stance work since the blog post of 2026-10-11. The walking pilot of §9
(decision D-D36, [BEHAVIOR_RECIPES_PLAN.md](../BEHAVIOR_RECIPES_PLAN.md) §6.2)
is pre-registered, **launched on Colab on 2026-10-17 and not yet judged**; its
outcome will be appended to this note as a dated note when it is judged. The
evidence, frozen copies of the scripts that decided each verdict, and their
sums are in [stance_2026_10/](stance_2026_10/README.md).

*Status 2026-10-18 (appended): the walking pilot was judged on 2026-10-17
and read STILL on all three seeds, final. §15, appended on 2026-10-18,
records that outcome, the offline, report-only certificates, battery, foot
and ankle audit and literature report of 2026-10-17, and the maintainer's
decision of 2026-10-17 to run a second pilot (pre-registered, not yet
trained, no result); their evidence was added to
[stance_2026_10/](stance_2026_10/README.md) on 2026-10-18. The status above
is unchanged.*

*Status 2026-10-18, later (appended): §16, appended the same day, records
the offline diagnosis of 2026-10-18 and the maintainer's decision D-D37 of
that day: three full T. rex stance screen runs on physics r9 instead of
walking pilot 2, which is parked; the runs were launched that day and are
not yet judged. Their criteria, Colab steps, recommendation and frozen judge
are in [stance_2026_10/stance3/](stance_2026_10/README.md). The statuses
above are unchanged.*

*Status 2026-10-19 (appended): §17, appended that day, records the judging of
the three stance screen runs by their fixed criteria: all three fail, the
decision table's line is 3b, and the best final checkpoint by the fixed rule
is Run 2's; with the report-only readouts, an overnight scratch experiment
(report only) and the maintainer's pending decisions, listed as proposals.
Their evidence is in
[stance_2026_10/stance3/results/](stance_2026_10/stance3/results/README.md).
The statuses above are unchanged.*

*Status 2026-10-19, later (appended): §18, appended that day, records the
maintainer's decision D-D39: the stance4 screen, another T. rex stance
screen, of three runs with a foot-stagger penalty, from scratch, judged
against two of §16's runs by criteria fixed before training (in
[stance_2026_10/stance4/](stance_2026_10/README.md)); not yet trained. The
statuses above are unchanged.*

*Status 2026-10-20 (appended): §19, appended that day, records the judging of
the stance4 screen by its fixed criteria: all three runs fail, the decision
table's line is 5, final, and the stagger penalty is not adopted; the best
stagger final is one episode short of stance3 Run 2's final, which stays the
walking pilot 2 parent the criteria offer; with the report-only extras, the
notebook's own gates, the diagnostic panels and the readouts, and two
overnight scratch experiments, report only: a replicate study of the stagger
penalty and a continuation test of a per-foot CoP band prototype. Their
evidence is in
[stance_2026_10/stance4/results/](stance_2026_10/stance4/results/README.md)
and the same folder's `night/`, `night2/` and `copband/`. The statuses above
are unchanged.*

*Status 2026-10-20, later (appended): §20, appended that day, records the
maintainer's decisions of 2026-10-20 that followed §19: D-D40, a from-scratch
screen of a per-foot centre-of-pressure band on the stagger arm, with the
stance4 runs as its control (its code, criteria, judge and Colab steps not yet
committed anywhere public); D-D39 amended with its outcome and the L4 runtime
confirmed; and D-D36 amended: walking pilot 2 is to be rebuilt for stance3 Run
2's final as its parent. The statuses above are unchanged.*

*Status 2026-10-22 (appended): §21, appended that day, records the stance5
screen of D-D40 judged by its fixed criteria: Run 1 (seed 52) holds, Run 2
(seed 60) fails K3 only and Run 3 (seed 42) fails K1 and K3; the decision
table's line is 3, final, so the band is not adopted, and the line asks for two
more seeds; s52's final ties stance3 Run 2's final under the parent rule. With
it: the band's build and package (the screen branch pushed at `cf80ecb1`, which
supersedes §20's "not committed anywhere public"), the report-only extras,
gates, certificates, panels and band readouts, the saturation alarm on seed 42
(`tail_3_pitch_act` at +1), the offline diagnosis of stance4 s42's saturation,
and walking pilot 2's rebuild (its branch pushed; not yet trained). Their
evidence is in
[stance_2026_10/stance5/results/](stance_2026_10/stance5/results/README.md) and
the same folder's `stance5/`, `stance4/satdiag/` and `pilot2/`. The statuses
above are unchanged.*

*Status 2026-10-22, later (appended): §22, appended that day, records the
maintainer's decisions of 2026-10-22 that followed §21: D-D41, a replication of
the unchanged band recipe on two more seeds (70 and 80), judged against the
stance4 control by new criteria fixed before training and read under a new rule
(the band is adopted only if both new seeds hold), with walking pilot 2's parent
kept for now; and the amendments of D-D40 (its build, runtime and outcome) and
D-D36 (walking pilot 2 rebuilt). The replication is not yet trained. The
statuses above are unchanged.*

*Status 2026-10-24 (appended): §23, appended that day, records the stance5
replication of D-D41 judged by its fixed criteria: Run 4 (seed 70) holds and
Run 5 (seed 80) fails K1 only, on the actuator-saturation rail (its alarm at
4,456,448, `tail_3_pitch_act` at +1); H = 2 of 5 and the decision table's line
is 2, final, so the band is not adopted; s52's final still ties stance3 Run 2's
final under the parent rule. With it: the report-only notebook gates,
certificates and extras, the two report-only workflows of §22.2 (the K3
diagnosis and the scratch tail experiment), the report-only K3 readouts added
on 2026-10-22, and the maintainer's decisions of 2026-10-24 (D-D42: the next
round targets the tail saturation habit, its design and criteria to be fixed
and approved before any training). Their evidence is in
[stance_2026_10/stance5/replication/results/](stance_2026_10/stance5/replication/results/README.md)
and the same folder's `stance5/replication/`, `stance5/k3diag/` and
`stance5/tailexp/`. The statuses above are unchanged.*

It continues [STANCE_REWARD_PILOT_FINDINGS_2026_10.md](STANCE_REWARD_PILOT_FINDINGS_2026_10.md)
(2026-10-10), [STANCE_HACK_AUDIT_2026_10.md](STANCE_HACK_AUDIT_2026_10.md)
§12 (the physics-r9 statue) and the blog post
[Teaching dinosaurs to stand](../../website/blog/2026-10-11-teaching-dinosaurs-to-stand/index.md).

## Short answer

- **No stance reward term reads where the weight sits fore and aft.** None
  reads contact positions or the centre of pressure (CoP). On the production
  reward, the held mean action of run A (the 12M physics-r9 stance run
  `20261011_165957`) earned 3758.9, 3766.8 and 3762.4 per episode scaled
  x0.1, x0.5 and x0.9 (right pad CoP 0.275, 0.623 and 0.940; only x0.9 is
  past the 0.80 bar), against 3755.7 for the statue (the zero-action policy)
  (§3.3): the lean was free. Under arm L's settled leg target a symmetric
  forward lean costs about 109 per episode per unit of CoP on identical
  postures, on physics r9 (`main`'s plant) and r10 (draft #623's foot) alike,
  and the r10 seeds still drifted onto their toes and earned less for it (§7).
  No term pays for the drift, and the restoring slope did not stop it.
- **Symmetry removed most of the lopsidedness, not the forward lean.**
  Mirror-symmetric PPO with a lateral lean term and a settled leg target
  (arm L) trained policies whose joint groups are mirror-symmetric to under
  0.1° (median), with no falls on the 120 final panel episodes and
  `stance_quality/v2` passed on two seeds of three. It failed the
  pre-registered fore-aft criteria (a forward creep: seed 42 with 9, 8 and 11
  episodes over the pad CoP bar at 4M, 5M and 6M, which the verdict classed
  as one-pad creep at 4M and 5M and mirror-even at 6M; seed 60 mirror-even,
  with 8 and 39 at 4M and 5M and 0 at 6M) and the joint-noise criterion (8 of
  60 falls against 7). With its fore-aft arm C also failing, the screen's
  verdict was O9 (both arms fail: no PR, no 12M run; §5).
- **On the r10 foot, arm L's forward lean was worse on 2 of 3 seeds.** With
  load-bearing digits (draft #623), arm L trained from scratch put seeds 52
  and 60 on their toes: 37-40 of their 40 panel episodes full-horizon and
  over the whole-foot CoP rail at each of 4M, 5M and 6M, and through 12M in
  training. All four fixed criteria failed (§6). On r10 a held forward lean
  that most spawns survive reaches CoP 0.89-0.94, against 0.76-0.79 on r9:
  falls no longer cut the drift off before the 0.80 rail (§7). The r9 and r10
  runs also differ in harness (§6.4). #623 is parked; `main` keeps physics
  r9.
- **Why the lean starts is not explained.** All three r10 seeds leaned
  forward by the first evaluation (whole-foot CoP 0.18, 0.33 and 0.32 at
  0.125M steps, against a held statue's 0.11). The explanation offered first,
  that a forward lean pays under exploration noise on r10, did not replicate
  on new spawns and is withdrawn (§11).
- **Capping the legs' authority held the fore-aft edge but failed on lateral
  pad roll and a slow toe-ward creep**, and was dropped (§4).
- **What is in the repository from this work:** report-only telemetry (#620
  per-foot pad CoP and leg DC; #621 gate progress across a same-stage resume;
  #622 whole-foot CoP; #626 `progress.json`, the roll rails, termination
  causes and the `PostureDrift` warning), the symmetry module and two T. rex
  stance options, off by default (#624), and the recipe label `recover`
  (#625, D-D35). Their landing records are in the status table of
  [CONSOLIDATION_PLAN_2026_09.md](../CONSOLIDATION_PLAN_2026_09.md) and in
  [CHANGELOG.md](../../CHANGELOG.md) (#626's added with this note).
  Not on `main` before this note: draft #623, the screens' criteria, judges
  and verdicts (copied into [stance_2026_10/](stance_2026_10/README.md) by
  this note) and the walking pilot's branch.
- **Next:** one short walking pilot from r9 arm L seed 60 at 6M, judged by
  hop-detecting criteria fixed before training plus a video review (§9;
  decision D-D36 of 2026-10-17,
  [BEHAVIOR_RECIPES_PLAN.md](../BEHAVIOR_RECIPES_PLAN.md) §6.2; launched that
  day). By the maintainer's choice of 2026-10-16, no more stance-only reward
  experiments unless walking points back to stance.

## 1. Scope and conventions

- Model `ModelId("tyrannosaurus_rex", "anatomical")`, stance stage. Run A
  trained on `cdffea89` (physics r9, the 64-wide observation before the
  heading-free migration, D-D31); every later run used physics r9 or r10 and
  policy interface r15 (63 wide).
- Panel: `stance_quality/v2`, 40 deterministic episodes on reset seeds
  3042-3081, at the panel's single heading. "Clean" is an episode that meets
  every episode criterion, so "29/40" means 29 clean. A v2 PASS needs at
  least 37/40 clean (exact one-sided 95% lower bound, LCB, at least 0.80),
  full horizon on at least 0.95 and at most one hop-or-fall episode. A
  full-horizon episode reaches the 1,000-step horizon; a hop-or-fall episode
  ends early or fails a support, touchdown, drift or command-saturation
  criterion (the gate's hop-or-fall keys). The
  stage's six-heading certificate (`stance_quality_heading/v1`, D-D32) was
  not part of any screen.
- Steps: in the screens M = 1,048,576 steps (4M = 4,194,304, 6M =
  6,291,456). The stage budget is 12,001,280 steps ("12M").
- CoP, + toe-ward. The pad CoP (`max_sole_cop_fore_aft`, bar 0.80) is in
  pad half-lengths; the r9 pad spans 0.070-0.210 m ahead of the ankle. The
  whole-foot CoP (#622, `max_foot_cop_fore_aft`) is the floor load's moment
  about the ankle over the limb's floor force, in half-lengths of the foot's
  support span; on r10 the span is [0.060, 0.213] m (half-length 76.5 mm)
  and the rail is 0.80. §7 reads both plants on the r10 span.
- Noise tests (H4, K3): 20 spawns per seed (3042-3061). "Joint-space" adds
  the frozen per-actuator sigma vector of the production control (mean
  0.161) to the joint targets; "own-sigma" samples the policy's own std.
- The statue is the zero-action policy. "(inference)" marks a claim not
  measured directly.
- The source files quoted here name paths of the session's scratch tree
  (`foot/...`, `scratchpad/...`). Only the files copied into
  [stance_2026_10/](stance_2026_10/README.md) are kept; its `SOURCES.txt`
  maps each one to its source.

## 2. Timeline

All times UTC.

| When | What | Evidence |
|---|---|---|
| 2026-10-12 10:34 | Run A (`20261011_165957`, started 2026-10-11 16:59, so the blog post of 2026-10-11 predates its verdict; seed 56, 12,001,280 steps, 17 h 23 min) finishes and is refused by `stance_quality/v2`: 1/40 clean, the pad CoP rail failing on 39/40 | `run_a/stance_gate_report_20261011_165957.txt`; the run's `stage_summary.txt` and `gate_verdict.json` on Drive |
| 2026-10-12 | Foot and CoP investigation: mechanism, reward, per-species anatomy, three 1M probes, two independent verifiers; report at 15:38 with 11 corrections | `run_a/report.md` |
| 2026-10-12 | Offline noise-ranking check of candidate terms (no training): no candidate passes | `noise_rank/` |
| 2026-10-12 | The maintainer approves per-foot pad telemetry, report-only (became #620) | CHANGELOG |
| 2026-10-12/13 | 1M screen: control, lower noise, lower noise plus a leg-asymmetry term, 5 seeds; both arms fail | `cap_screens/screen_1m_analysis.txt` |
| 2026-10-13 | 3M noise screen (fail); 3M screen of a CoP term and reduced sagittal leg authority; 6M authority extension (fail); lateral pad-roll diagnosis without training, independently re-measured | `cap_screens/` |
| 2026-10-14 | Hip-roll screen (authority plus hip roll x0.01) fails at 6M; the maintainer chooses "Symmetry, no caps"; mirror map, equivariance, lean-term calibration and screen design (v1 to final r2), no training | `cap_screens/screen_hip_roll_analysis.txt`, `symmetry/` |
| 2026-10-14 | #621 (gate progress across a same-stage resume, a bug the 6M screen found) | CHANGELOG |
| 2026-10-15 01:00 | READY-v1: design, parameters, verdict scripts and bars hashed before the first training step | `symmetry/frozen_params.json` |
| 2026-10-15 02:35, 17:57 | The cloud container is reclaimed (idle session) and later restarts; two seed-segments per event re-run once under the predeclared rule | `symmetry/rerun_ledger.json` |
| 2026-10-15 11:58 / 20:44 | Symmetry screen interim (both arms continue) / verdict O9 (both arms fail) | `symmetry/interim.json`, `symmetry/verdict.json` |
| 2026-10-15/16 | The maintainer chooses foot anatomy first; foot proposal (three variants, two review rounds), recommending the V3 foot as physics r10 | #623 body |
| 2026-10-16 01:24-06:54 | The maintainer's gut check ("a pretty deep rabbit hole"); at 06:54 the maintainer chooses the simplified plan: one fixed-criteria screen of the new foot, then walking, and no more stance-only reward experiments unless walking points back to stance | session record |
| 2026-10-16 | Whole-foot CoP telemetry approved (became #622); the symmetry module and the two options approved, off by default (became #624); draft #623 opens at 11:06 | CHANGELOG, #623 |
| 2026-10-16 13:59 | r10 screen criteria K1-K4 written; tie-break clarified at 16:30; the maintainer starts the first Colab run at 17:47 | `r10_screen/criteria.txt` |
| 2026-10-16 | Recipe label `recover` for the recovery stage (D-D35, #625) | CHANGELOG |
| 2026-10-16 20:25 | `gate_progress.npz` read by hand from Drive: seeds 52 and 60 on their toes at 2.3-2.5M; the runs are left alone | §6.4 |
| 2026-10-16 22:33 | The maintainer asks for side work: the reporting change (became #626), walking-pilot preparation, the toe-lean diagnosis | session record |
| 2026-10-17 00:30-04:11 | Toe-lean diagnosis v1, v2, v3; pilot criteria verified in rounds at 01:22, 03:07 and 04:11 | `diagnosis/`, `pilot/criteria.txt` |
| 2026-10-17 04:11-12:38 | The organisation's monthly spend limit stops the agents; the round-3 verifications of the diagnosis and of the reporting change fail to run | session record |
| 2026-10-17 05:24-05:48 | The three r10 runs finish 12,001,280 steps | `r10_screen/report_only_summary.json` |
| 2026-10-17 12:46-13:20 | Nine r10 checkpoints judged and scored (FAIL on K1-K4), then independently re-derived | `r10_screen/` |
| 2026-10-17 13:21 | The maintainer parks #623 and chooses the pilot's parent, r9 arm L seed 60 at 6M (decision D-D36 of 2026-10-17) | `pilot/criteria.txt`; D-D36 in [BEHAVIOR_RECIPES_PLAN.md](../BEHAVIOR_RECIPES_PLAN.md) §6.2 |
| 2026-10-17 13:22 | Screen result posted on #623 | [comment](https://github.com/kuds/mesozoic-labs/pull/623#issuecomment-6234386057) |
| 2026-10-17 13:32-14:46 | Duty-guard grids; round-4 verification (no blocking item); fourth wording revision; branch `claude/pilot-walk-r9` pushed at `eaeef50f`, no PR | `pilot/` |
| 2026-10-17 | `main` at `1b9be186`, which includes #626 | git |
| 2026-10-17 17:01-17:28 | The walking pilot is launched on Colab. The first launch (about 17:01-17:03) ran the chain loop and began training a stance from scratch; that it was Runtime > Run all with the manual cell at its defaults is an inference from the notebook state (cells run in order to the chain loop, `MANUAL_NODE` None), not confirmed. A Drive check caught it at 17:06; the runtimes were stopped and the three run folders deleted, and the seeds were relaunched under the same names: `pilot_walk_s42`, `_s52` and `_s60` created at 17:13, 17:23 and 17:24, each holding the run-level files (`provenance.json`, `trunk_run.json`, `zero_action_baseline.json`) and a `03_locomotion` stage (created 17:17, 17:26 and 17:28), no `01_stance` | session record; Drive |

## 3. Run A: the right pad on its toe edge

Sources: `run_a/report.md` (the corrected report of 2026-10-12) and the
files it names, copied into `run_a/`.

### 3.1 The verdict

Run `20261011_165957`, seed 56, commit `cdffea89`, 12,001,280 steps. The gate
judged `robust_best_model` (evaluation 3770.1 ± 15.7 at 3.2M steps):

- 1/40 clean, LCB 0.0013.
- `max_sole_cop_fore_aft` (bar 0.80) fails on 39/40, median 0.9999.
- Sole tilt 4/40 (worst 2.85°), corner lift 3/40 (6.0 mm), sole contacts
  2/40, settle peak 1/40. All 40 reach the horizon; worst drift 1.6 cm.

### 3.2 Mechanism

**The pad is a two-station lever.** MuJoCo box-plane contact loads the
14 x 9 cm pad only at its heel corners (7.0 cm ahead of the ankle) and its
toe corners (21.0 cm). Every contact above 0.1 N sits on one of those two
lines (20,662 in the statue, 14,468 in the held run A action), so the pad CoP
reads +1 once the heel corners carry nothing. 92.4-93.6% of the statue's
digit load acts at the digit bases, inside the pad's heel band.

**The moment budget.** Whole-foot CoP is the ankle moment over the floor
force once the foot's weight (1.76 kg, 1.34 N·m) is included. At 420.5 N
the pad carries 29.3-88.2 N·m about the ankle. The statue uses 59.6 N·m
(CoP 14.2 cm ahead of the ankle): 28 N·m of toe-ward margin and 30 N·m of
heel-ward margin, 1.7-1.9° of ankle servo error at 900 + 40 N·m/rad. Under
run A's held mean action the ankle carries 83-84 N·m and the CoP sits at
20.3-20.5 cm, on the toe-corner line of the 1° toe-down pad (20.55-20.62
cm). These are torque (servo-error) windows: an ankle target offset moves
its own pad by only -0.036 to -0.039 half-lengths per degree.

**The driver is a small left/right-antisymmetric leg command, not a body
lean.** Run A's mean action holds the left ankle at about -3.7° and the right
knee at about +2.0°. Held-pose ablation, 20 seeds, ramped in from step 200
(`run_a/ablate_table.txt`):

| Held action | Right pad signed CoP | CoP > 0.8 | Clean |
|---|---|---|---|
| statue | +0.183 | 0/20 | 20/20 |
| full mean action | +1.000 | 19/20 | 0/20 |
| hip pitch only | +0.073 | 0/20 | 20/20 |
| all but the left ankle | +0.292 | 0/20 | 17/20 |
| left ankle only | +0.883 | 7/20 (12 fell) | 1/20 |
| all but neck, head and tail | +1.000 | 19/20 | 0/20 |
| mirrored mean action | left pad +0.980 | 20/20 (left) | 0/20 |

Scaling the action from x0.1 to x0.9 moves the right gate CoP linearly from
0.275 to 0.940.

**Why a few degrees suffice: the closed chain.** With both feet planted one
ankle or knee target offset moves the *other* foot's CoP by -0.19 to -0.23
half-lengths per degree, and its own by -0.04 (ankle) or +0.11 (knee)
(`run_a/report.md` §2). Ramped in after the settle, one ankle offset brings
the other pad to 0.80 at -3.1 to -3.25° (right ankle) and -3.5° (left)
(`run_a/thresholds.txt`). Both Compsognathus models couple the feet as
strongly (anatomical: other foot -0.22 and -0.39 per degree of ankle and knee;
robot: +0.22 per degree of ankle, opposite in sign, not examined), so this is
a property of two position-servoed legs in double support, not of the T. rex
foot (inference for robots in general).

**When it formed.** Run A's left-minus-right ankle DC (the mean commanded
action, in action units) went from +0.007 (before 0.35M) to -0.049 by
0.85-1.0M and stayed at -0.048 to -0.066 to 12M: the edge posture was in
place about 11M steps before the verdict. On `main` both 1M probe seeds
formed the edge between 0.5M and 1M, so 1M screens are informative.

**Ruled out.** Physics r9 (the same front-edge pad appeared on r8; the r9
statue is 39/40 clean). The policy interface (the heading-free observation
reproduces it at 1M). A T. rex-specific foot (the coupling above). Toe
compliance (with toes 50x stiffer the held action still puts the right pad at
0.924, 17/20 above 0.8). The neck, head and tail (without them 19/20 episodes
are still on the edge).

### 3.3 Why the reward allowed it

- **No term reads CoP.** The foot terms read summed pad and digit touch,
  pad tilt and the foot sites; nothing reads contact positions.
- **The return is flat in CoP.** The statue earns 3755.7 per episode, 3,081.7
  (82%) of it from seven saturated terms. Run A's action scaled x0.1 / x0.5 /
  x0.9 earns 3758.9 / 3766.8 / 3762.4 while the right CoP goes 0.275 → 0.623
  → 0.940. Run A's legs minus a hips-only posture: -3.41 ± 4.26 per episode
  (n = 28) for gate CoP 0.941 against 0.218 (`run_a/reward_pairs.txt`).
  foot_flatness (-5.9), foot_load_balance (-4.2) and nosedive (-1.3) are
  cancelled by leg_home_pose (+3.4), posture (+1.9), height (+1.5) and
  tail_home_pose (+0.8).
- **Training noise hides the deterministic posture.** At the trained
  policies' own σ of 0.16 the statue falls in 5/20 episodes, and at 0.19 in
  20/20. With the same noise draws the statue earns 438-2,251 less per
  episode than the trained policies, yet without noise the policies earn
  10.6-29.6 less than the statue (`run_a/noise_s48.txt`, `noise_s56.txt`).
  With the policy's noise added, the statue's per-step pad CoP is past 0.8 on
  96% of steps, and a clipped per-step CoP cost reads 0.951 for the statue
  against 0.955-0.956 for the trained edge policies: such a term saturates
  (`run_a/noise_saturation.txt`). (inference) PPO's return mostly prices
  surviving the noise.
- **Observability** (inference, `run_a/observability.txt`): the
  `foot_contact` channels carry no CoP information (R² ≤ 0.13); the passive
  toe angles predict pad CoP on held-out postures (rms 0.07-0.29
  half-lengths).

### 3.4 Probes, 1M steps on `main` (`8565d299`)

Final checkpoints, panel 3042-3081 (`run_a/probe*_*.txt`):

| Arm | Seed 48: clean (CoP > 0.8) | Seed 56: clean (CoP > 0.8) |
|---|---|---|
| control (`main` as is) | 21 (19) | 9 (31; one fell) |
| per-step CoP margin, w 0.10, safe 0.65 | 12 (26) | 1 (38) |
| `leg_home_pose` centred on the settled pose | 28 (8) | 21 (18) |
| statue | 39 (1) | |

The per-step CoP term that helped on r8 (62 → 21 of 120, the pilot
findings §1.1) did not help here; its cost was flat at -0.075 per step from
0.26M. The settled-reference arm's reduction lies inside the control's own
seed spread and was trained on a different driver, so it is not attributed
to the arm. Reward-selected `robust_best` checkpoints were near-statue
policies at 50k-350k steps (34-39/40 clean) in the probes, and the edge
policy at 3.2M in run A: reward-based selection is not a safeguard.

### 3.5 Per-species foot review

`run_a/report.md` §4, `run_a/anatomy_summary.txt` and
`run_a/statue_per_species.txt` hold the statue panels of every model. In
short: the T. rex statue's pad takes 82.7% of the load and stands at pad CoP
0.287 [0.10-0.48] against the 0.80 bar, on a lateral pad edge (outer CoP
0.954, not gated); the Compsognathus robot has no sole bars, so an edge
stance would pass its gate; the Dibothrosuchus statue sits at CoP 0.745 on
the heel side, 0.31° of ankle offset from 0.80, while the trackways the model
stands for are digitigrade (§13). None of these was a fix for run A.

### 3.6 What followed in the repository

The report's first recommendation became #620: per-foot signed pad CoP,
digit share and the left-minus-right sagittal leg DC (in degrees about the
leftward axis) at every evaluation, report-only. The whole-foot CoP became
#622, report-only, with no stage declaring its key on `main`.

## 4. Before symmetry: noise, a CoP term and capped leg authority

### 4.1 Offline noise-ranking check (2026-10-12, no training)

Each candidate anti-edge term was scored per step on held postures (statue,
hips-only, run A's action scaled and held) under training-shaped noise, with
a pass criterion fixed in the report. **No candidate passed**
(`noise_rank/summary.txt`, which supersedes the first version). The
unbounded left-right leg quadratic rewarded falling sooner (worst gain from falling sooner 177 at uniform σ 0.22,
against 14.5 for the base reward alone); an independent verifier found it
(`noise_rank/verify_summary.txt`). Its bounded replacement, `d_ema50_sat5`
(an EMA over 0.5 s of the left-minus-right hip pitch, knee and ankle, w 0.03
per deg², at most 0.15 per step), was chosen by a rule written before the
sweep on panels A and B and validated on held-out panels C and D; one width
condition (E5) was added after the first sweep run, and the record says so.
Under policy-shaped noise it kept the gap at σ 0.16 but reached the
20-per-episode gap at σ 0.22 on 1 of 4 panels.

### 4.2 The screens

Every screen judged fixed checkpoints on the panel, with criteria written
before it trained.

| Screen | Arms (seeds; budget) | Result against its criteria |
|---|---|---|
| 1M (2026-10-12/13) | control; `log_std_init` -2.5 ("lownoise"); lownoise plus `d_ema50_sat5` (42/48/52/56/60; 1,048,576) | The control reproduced the edge on 5 of 5 seeds (CoP > 0.8 on 21/19/28/30/12 of 40). Lower noise cut final CoP failures 110 → 15 of 200 (one-sided rank-sum on per-seed counts p = 0.004); the asymmetry term added nothing measurable (11 against 15, p = 0.42). Neither arm reached 37/40 on any seed at the final checkpoint: FAIL |
| 3M noise (2026-10-13) | `log_std_init` -2.5 and -2.0 (42/52/60; 3,145,728) | -2.5 delays the edge, CoP > 0.8 at 1M/2M/3M 4/10/2, 3/13/23, 16/19/22; -2.0 neither delays nor prevents it (40/32/40 at 3M). No seed reached 37/40 at 2M or 3M under any of the tilt and corner-lift options tried; the CoP rail binds. Its matched-noise comparison is withdrawn (§11) |
| 3M CoP term and authority (2026-10-13) | control to 3M; `cop`, a load-gated 0.5 s EMA pad-CoP centring bonus (w 0.28); `authority`, hip pitch x0.055 (±3.575°) and knee and ankle x0.025 (±1.25°) of the residual action (42/52/60; 3,145,728) | Control CoP > 0.8 39/37/37 at 3M. `cop` delayed the edge (35/120 at 1M against 60) but reached 99/120 at 3M against 113: dropped. `authority` had no full-horizon CoP failure through 3M (clean 38/33/38), but seed 42's right pad crept toe-ward (+0.045 ± 0.006 per M): extended to 6M |
| 6M authority extension (2026-10-13) | `authority` resumed to 6,291,456 | E1 pass (CoP > 0.8 0/0/0, 0/1/0, 0/0/2, all on falls). E2 FAIL: seed 42's right pad projected to 0.668 at 12M against 0.60 (series concave, quadratic t -4.3, kept as a fail). E3 pass (joint-space falls 3/60). E4 FAIL (clean 30/34/36). All 50 non-fall failures at 3-6M were lateral, 48 of them pad roll |
| Hip roll (2026-10-14) | `authority` plus hip roll x0.01 (42/52/60; fresh to 3M, resumed to 6M) | Interim pass (roll-rail 3/120). At 6M: H1 pass (40/36/38); H2 FAIL (seed 52 4/40 against a per-seed limit of 3; pooled 6/120, at its limit); H3 FAIL (12M projections 0.757 and 0.818 against 0.60); H4 pass (1/60); H5 pass (0.178°) |

### 4.3 Lateral pad roll (2026-10-13, no training)

On the authority arm's 15 full-horizon roll failures at 6M the failing pad's
window roll was 2.34° (median). 35% of it [25-73%] was set by the reset
(hip-roll reset splay held by friction; the statue on the same spawn shows
it) and 65% was added by the policy: 49% a sideways sway from the hip-roll
common mode (1.42° commanded, moving the pelvis 18.7 mm over the feet while
pelvis roll stayed within 0.2°) and 16% stance narrowing. 47 of 48 roll
failures at 3-6M were on the 15 most-splayed of the 40 spawns. Proxies on
the full panel, without training: clean 30/34/36 as trained, 38/39/38 with
hip roll zeroed, 38/40/39 with its common mode removed and 39/40/40 with a
nominal-width reset (`cap_screens/roll_diag_analysis.txt`). An independent
re-measurement reproduced the base and statue episodes to 1e-14 and refuted
no headline claim; it corrected one (§11). Of the options, (a) bounding hip
roll was screened (the hip-roll screen, above), (b) a bounded lateral lean
term became arm L's lean term, (c) lower hip-roll reset noise is a
task-design decision left to the maintainer, and (d) a statue-relative roll
rail was not recommended.

### 4.4 Outcome

Neither capped arm passed its predeclared criteria, so the spec of a
capped-authority policy interface (r16) and its 12M confirmation stay on
file, not triggered. The hip-roll cap was safe for balance (panel falls 4 →
0 of 120 against the authority arm) but did not bound the toe-ward creep on
2 of 3 seeds. The maintainer chose "Symmetry, no caps" on 2026-10-14.

## 5. Symmetry: arms L and C, verdict O9

### 5.1 Groundwork without training (2026-10-14)

- **The asymmetry carried the failures.** Averaging each stored checkpoint
  with its mirror image (π_sym, no retraining) took the 3M control from 0/2/1
  to 37/30/30 clean and from 113 to 18 of 120 CoP failures, and the hip-roll
  arm from 40/36/38 to 40/40/39 clean. What symmetry cannot remove is a
  mirror-even forward CoP: π_sym of the control kept 10 and 6 CoP failures on
  seeds 52 and 60, on both pads. An independent re-run reproduced the π_sym
  judges bit for bit.
- **The mirror-even creep is a learned forward lean.** The centre of mass
  sits 18-45 mm ahead of the statue's, from a common-mode leg command (both
  ankle targets -2.4 to -3.5°); the even pad CoP tracks it at 0.0129
  half-lengths per mm (r 0.975). About half of the incentive was
  `leg_home_pose` pulling toward the authored keyframe, which the statue
  sags off (+18.3 per window at the deterministic optimum). This motivated
  re-centring the leg target on the settled pose and a fore-aft term.
- **Method: a KL-form mirror loss, not mirror augmentation.** On the 3M
  control's own rollouts 99.7-100% of augmented mirrored rows fall outside
  PPO's clip band (per seed), and at an asymmetry of 0.03 still 80-91%
  (`symmetry/mirror_ratio.json`); the corrective signal vanishes at the
  stance's small exploration noise. `MirrorKLPPO` adds kappa times a KL-form
  mirror loss whose gradient is linear in the asymmetry.
- **The lean term** (`symmetry/lean_calibration.json`): r = 0.25
  exp(-(d / 15 mm)²), d the whole-body CoM's lateral offset from the foot-site
  midpoint, paid at steps 200-999 with both feet supported. The weight is
  2 / (statue cost at w = 1) = 0.2885 rounded down; the statue's cost is 1.73
  per episode. Its window cost separates roll-rail episodes from the rest
  with AUC 0.986 (pooled), and it is mirror invariant to 1.5e-13 mm.
- **The mirror map is exact.** The T. rex model mirrors to 0.0 m; over 100
  paired mirrored rollouts (25,065 steps) the step-1 observation error is at
  most 1.0e-11. The Compsognathus robot is not mirror-symmetric as authored
  (parts of its right leg sit 36.7-36.9 mm from the mirror of their left
  partners) and is refused by #624.

### 5.2 Design and pre-registered criteria

Frozen in READY-v1 (2026-10-15 01:00:53, before the first training step;
`symmetry/frozen_params.json`, whose original sha256 `58b2c8fb…` READY-v1
recorded). Two arms, seeds 42/52/60, fresh to 3,145,728 with an interim
judgement, then resumed to 6,291,456; at most two heavy processes.

- **Arm L** (`sym_lean_settled`): `MirrorKLPPO` (kappa 0.3, sigma floor
  0.01, mirrored critic, symmetric observation statistics, tied `log_std`)
  plus the lateral lean term (0.25, 15 mm) plus `leg_home_pose` centred on
  the mirror-symmetrised settled statue pose.
- **Arm C** (`sym_lean_settled_fa`): arm L plus an instantaneous fore-aft
  CoM term, 0.375 exp(-(max(|d - d_ref| - 7.5 mm, 0) / 15 mm)²) after step 300,
  d_ref -16.09 mm.
- Interim (3M): I1 pooled roll-rail ≤ 5/120; I2 falls ≤ 2/120; I3 at most one
  seed with more than 3/40 CoP > 0.8.
- 6M: H1 v2 PASS on ≥ 2 of 3 seeds; H2 roll-rail ≤ 6/120 pooled and ≤ 3/40 per
  seed; H3 (a) CoP > 0.8 ≤ 3/40 on every seed at 4M, 5M and 6M and (b) each
  pad's 3-6M linear fit projected to 12M ≤ 0.60; H4 joint-space falls ≤ 7/60
  and own-sigma falls ≤ 3/60; H5 pooled median policy-added roll ≤ 0.30° on
  the focus spawns; H6 6M panel falls ≤ 2/120; S1 and S2 symmetry checks
  (S1: per joint group, the 6M mirror asymmetry within 0.1° for hip roll and
  0.3° for the others at the median and four times that at the 90th
  percentile, and the mean antisymmetric command within the same bounds; S2:
  replacing the policy by its mirror average π_sym changes at most 8
  episodes' clean labels and the clean count by at most 5).
- Outcomes O1-O9 in a total order; O9 = both arms fail: no PR, no 12M run.

Both the lean term and the re-centring are in both arms, so the screen cannot
say whether either is needed.

### 5.3 Interim at 3M (2026-10-15 11:58)

Both arms continued. L: roll-rail 3/0/0, falls 0, CoP > 0.8 6/2/0; seed 60
passed v2 at 40/40. C: zero on all three. An independent recount agreed on
all 120 quantities (`symmetry/J1_recount.log`).

### 5.4 Verdict at 6M (2026-10-15 20:44): O9, both arms fail

Per seed, clean / 40 (CoP > 0.8 episodes, all 40 counted), from
`symmetry/verdict.json`:

| Arm, seed | 3M | 4M | 5M | 6M |
|---|---|---|---|---|
| L 42 | 30 (6) | 30 (9) | 32 (8) | 29 (11) |
| L 52 | 35 (2) | 36 (3) | 39 (1) PASS | 38 (2) PASS |
| L 60 | 40 (0) PASS | 31 (8) | 1 (39) | 40 (0) PASS |
| C 42 | 7 (0) | 19 (2) | 8 (0) | 39 (0) PASS |
| C 52 | 1 (0) | 1 (1) | 31 (1) | 4 (22) |
| C 60 | 10 (0) | 2 (2) | 0 (0) | 38 (2), 2 falls |

- **Arm L:** H1 pass (seeds 52 and 60), H2 pass (roll-rail 2/0/0), H3a FAIL
  (9/8/11, 3/1/2, 8/39/0), H3b FAIL (seed 42 right pad 0.656; seed 60 0.931
  and 1.173), H4 FAIL (joint-space falls 1 + 1 + 6 = 8/60; own-sigma 0/60),
  H5 pass (0.227°), H6 pass (0 falls), S1 and S2 pass. Failed families: COP
  and ROBUSTNESS.
- **Arm C:** nearly no front-edge failures early (0 on every seed at 3M), but
  `max_actuator_saturation_fraction` failed on 32/39/29 of 40 episodes at 3M
  (1/34/2 at 6M), and seed 52 crept onto one pad at 6M (22 CoP failures). H1
  FAIL (only seed 42), H3 FAIL, H5 FAIL (0.391°); H2, H4 and H6 pass. Failed
  families: COP, ROBUSTNESS and ROLL. (Which actuators saturated is on file
  only as narrative: the tail tip and, on seed 42, the head pinned as a
  counterweight; unverified.)
- **Symmetry worked as a method.** Every arm L joint group's median mirror
  asymmetry at 6M was under 0.1° (largest 0.093°, hip pitch on seed 60),
  against 3.7-7.6° of applied joint-target asymmetry in the 3M control.
- The two independent verdict implementations agreed on all 537 quantities
  (`symmetry/J2_recount.log`), so the arbiter was not needed.
- Validity: no method failure, implementation defect, collapse stop or
  imputed checkpoint. The two infrastructure events each used the one
  predeclared re-run per seed-segment (`symmetry/rerun_ledger.json`).

### 5.5 What it rules in and out

- In: symmetric training removed most of the left/right antisymmetric part
  of the stance failures (arm L's joint groups mirror-symmetric to under
  0.1°; roll-rail 2/0/0 at 6M), but a symmetric policy can still creep onto
  one pad: the verdict classed arm L seed 42 at 4M and 5M (pad bias 0.20 and
  0.17), arm C seed 42 at 4M (0.16, 2 CoP failures) and arm C seed 52 at 6M
  (0.43, 22 CoP failures) as one-pad creep.
- Out: symmetry alone does not stop a mirror-even forward creep; arm L's
  seed 60 went from 0 to 39 and back to 0 CoP failures between 3M and 6M, so
  a single checkpoint says little.
- Not answered: whether the lean term and the re-centring are needed (both
  arms had them), and how much of the arms' behaviour comes from extra
  optimisation (the symmetry arms ran 9.46-9.76 Adam epochs per update over
  0-3M, against the control's cumulative 7.13-7.52 at 3M; the in-training
  flags F1, the rollout mirror asymmetry above 0.1° of hip roll or 0.3°
  elsewhere for 16 updates, and F6, a 1M window's mean Adam epochs per update
  more than 1.5 away from the control's, fired in all six fresh segments and
  are record-only).
- The predeclared follow-ups on a failure (kappa x3, the E2 fallback of
  dual-clip mirror augmentation plus the KL loss, a recalibrated fore-aft
  term) were not taken up: the maintainer chose foot anatomy first.

### 5.6 What landed

#624 adds `MirrorKLPPO` (`[ppo] symmetry = "mirror_kl"`), the T. rex
`lateral_lean_weight` / `lateral_lean_tolerance_m` and
`leg_home_pose_reference = "settled"`, all off by default; no stage config
enables them ([mirror-symmetry.md](../../website/docs/training/mirror-symmetry.md)).
The screen above ran the scratch equivalents of these switches; the r10
screen (§6) ran #624's.

## 6. The r10 foot and the r10 + arm L screen

### 6.1 The foot (draft #623, D-D34, parked)

From the foot proposal of 2026-10-15/16 (three variants, two review rounds;
the evidence is summarised in the #623 body):

- The 14 x 9 cm plantar pad becomes a 10.5 x 11 cm sole box, 0.060-0.165 m
  ahead of the ankle. Digits II/III/IV reach 0.171/0.212/0.178 m (Sue's
  ratios, the top of the plausible range) and splay 25° and 33° off digit
  III. MTP springs go from 40 to 3000/4500/3000 N·m/rad.
- The T. rex stance rail becomes `max_foot_cop_fore_aft = 0.80` on the span
  [0.060, 0.213] m (the toe line from a held-offset rule); the pad CoP becomes
  report-only on the T. rex.
- Physics r9 → r10; the policy interface stays r15.
- Margins against r9: forward push limit 58.4 against 55.4 N (+5.4%); the
  first nosedive under a held single-ankle offset 1.0-1.5° later; under the
  H4 joint-noise vectors, four r9-trained policies fall 0/80 against 9/80
  (Fisher p = 0.003).

### 6.2 Zero-shot evidence did not predict training

r9-trained arm L checkpoints read zero-shot on r10 looked better: episodes
over 0.80 at 4M/5M/6M of seed 42 5/1/5 (16/11/18 on r9, as trained), seed 52
1/0/1 (3/1/2), seed 60 9/30/0 (19/40/1), and H4 joint-noise falls 3/2/0 of 60
against 10/24/8 (the #623 body, one common whole-foot CoP statistic on each
plant's own held-offset line; not the pad-CoP counts of §5.4 or K2). The PR
body said that only training on r10 could show whether a policy trained there
avoids the forward lean, and held the PR for that.

### 6.3 Criteria, fixed before training

Written 2026-10-16 13:59:44 (`r10_screen/criteria.txt`), about four hours
before the first run; branch `claude/screen-trex-r10-arm-l` at `026eb833`
(#623's foot plus a config turning arm L on), notebook, seeds 42/52/60,
`n_envs` 4, the stage's 12M budget, judged at 4M, 5M and 6M. All four must
hold:

- **K1:** v2 PASS at 6M on at least 2 of 3 seeds.
- **K2:** on every seed, at most 3 of 40 full-horizon panel episodes over the
  fore-aft rail (r10: `max_foot_cop_fore_aft` > 0.80), at each of 4M, 5M and
  6M.
- **K3:** at 6M, joint-space noise falls ≤ 7/60 pooled and own-sigma falls ≤
  3/60 pooled.
- **K4:** at 6M, full-horizon roll-rail failures (tilt > 2.0° or corner lift
  > 0.004 m) ≤ 6/120 pooled and ≤ 3/40 per seed.

If any fails, the walking pilot starts from the best of the six 6M
checkpoints (three r10, three r9): most clean episodes, a tie going to fewer
noise falls. The only later edit, at 16:30:25 and before any r10 checkpoint
existed, clarified "noise falls" as joint-space plus own-sigma summed; joint
falls alone pick the same checkpoint.

### 6.4 Result: FAIL on K1-K4

At 2.3-2.5M steps the in-training telemetry, decoded by hand from Drive,
already showed seeds 52 and 60 on their toes (whole-foot CoP 1.11 and about
1.25; digits carrying 70-90% of the load). The runs were left alone so the
fixed criteria stayed valid. Per checkpoint (`r10_screen/score.json`,
`r10_cells`; recomputed in `r10_screen/verify_recompute.json`):

| Seed, step | v2, clean (LCB) | Full-horizon over the rail | Roll-rail | Noise falls, joint / own | Reward (statue 3982.4) |
|---|---|---|---|---|---|
| 42, 4M | FAIL, 35 (0.755) | 5 | 1 | 0 / 0 | 3935.9 |
| 42, 5M | PASS, 38 (0.851) | 2 | 0 | 0 / 0 | 3929.5 |
| 42, 6M | PASS, 40 (0.928) | 0 | 0 | 0 / 0 | 3951.7 |
| 52, 4M | FAIL, 3 (0.021) | 37 | 9 | 7 / 0 | 3829.6 |
| 52, 5M | FAIL, 0 | 40 | 11 | 2 / 0 | 3834.9 |
| 52, 6M | FAIL, 0 | 38 of 38 (2 more ended in head contact) | 4 | 7 / 2 | 3683.8 |
| 60, 4M | FAIL, 0 | 39 of 39 (1 more fell) | 2 | 0 / 0 | 3756.5 |
| 60, 5M | FAIL, 0 | 40 | 1 | 4 / 0 | 3822.4 |
| 60, 6M | FAIL, 0 | 39 of 39 (1 more fell) | 8 | 9 / 0 | 3740.5 |

| Criterion | r10 + arm L | r9 + arm L (same tests, §5) |
|---|---|---|
| K1 | FAIL: 1 of 3 (clean 40/0/0) | pass: 2 of 3 (29/38/40) |
| K2 | FAIL: seed 42 5/2/0, seed 52 37/40/38, seed 60 39/40/39 | FAIL (pad rail, all 40 episodes counted as in H3a): 9/8/11, 3/1/2, 8/39/0 (full horizon only: 8/7/11, 3/1/2, 8/38/0) |
| K3 | FAIL: joint 0 + 7 + 9 = 16/60, own 2/60 | FAIL: joint 8/60, own 0/60 |
| K4 | FAIL: 0 + 4 + 8 = 12/120 | pass: 2/120 |

Whole-foot CoP means (signed, right / left): seed 52 0.848/0.959,
0.944/0.977, 1.004/1.009 at 4M/5M/6M; seed 60 1.055/1.085, 1.045/1.087,
0.976/1.003; seed 42 0.30-0.48.

The comparison with r9 is not of the foot alone. The r9 arm L runs trained
locally with the screen's scratch wrapper, fresh to 3M and then resumed to
6M; the r10 runs trained on Colab with the notebook and #624's options in
one 12M run. The environments were shown equivalent, the training side was
not (`diagnosis/findings_v3.txt` §6), and there are three seeds per plant.

### 6.5 Verification

An independent pass recomputed K1-K4 from the per-episode rows
(`r10_screen/verify_recompute.json`, LCB agreeing to about 1e-14) and
re-judged seed 52 at 6M (the cell closest to a bar) and seed 42 at 6M (the
best pick) with no field differences. All nine checkpoints bind to plant r10
(`physics_sha256 1a05bc2d…`), commit `026eb833`, the right task and seed; no
relaxing flag was used. Before use the judge had reproduced an r9 judge file
bit for bit and its scorer had passed 33 boundary cases.

### 6.6 Best stance by the fixed rule, and the parent decision

By the rule, the best of the six 6M checkpoints is **r10 arm L seed 42**
(40/40 clean, 0 + 0 noise falls), tied on clean with r9 arm L seed 60 (40/40,
6 + 0) and winning the tie-break. Clean is not like-for-like across plants:
r10 is judged on the whole-foot rail and r9 on the pad rail, and r10 seed 42
at 6M has pad CoP over 0.8 on 17/40 episodes (report-only on r10). On
2026-10-17 at 13:21 the maintainer parked #623 (`main` keeps physics r9) and,
departing from the rule, chose r9 arm L seed 60 at 6M as the walking pilot's
parent, because the rule's choice sits on the unmerged r10 foot (§9).

### 6.7 Report only: 12M and the second half of training

- The notebook's end-of-stage six-heading certificate judged
  `robust_best_model.zip`, chosen by evaluation reward (highest mean minus
  standard deviation), not the 12M policy: seed 42 PASS (37/40 seeds clean at
  all six headings, LCB 0.8174), seed 52 FAIL (35/40), seed 60 FAIL (36/40).
  By its Drive modification times that file was last written at about 4.3M
  steps on seed 42, about 262k on seed 52 and at the first evaluation
  (131,072) on seed 60. The certificate's `best_eval_timestep` (5,636,096,
  131,072 and 131,072) is the best evaluation by mean reward, a different
  selection.
- 6M-12M in training (44 evaluations of 10 episodes per seed,
  `r10_screen/report_only_summary.json`): seed 52 over the rail on 423 of 423
  full-horizon episodes, 17 falls; seed 60 on 371 of 371, 69 falls (10, 12,
  20, 12, 13 and 2 per 1M bin); seed 42 74 of 439 over the rail, its
  over-bar fraction rising from 0.035 over 0-6M to 0.169 over 6-12M.

## 7. Why seeds 52 and 60 stood on their toes (toe-lean diagnosis)

Analysis only, no checkpoint replayed (`diagnosis/findings_v3.txt`; v1 is
kept as `diagnosis/findings_v1_superseded.txt`). The probe holds symmetric
postures along the mean action DC of seeds 52 and 60 (lambda scales it) on
both plants, in the arm L environment as trained; CoP on the r10 span.

- **No reward term pulls the policies onto their toes.** On identical held
  legs-only postures that every spawn survives on both plants (lambda 0-0.5)
  the total-return slope is -108.9 [-113.8, -104.5] (r10) and -108.6
  [-114.1, -104.0] (r9) per unit whole-foot CoP per episode;
  `leg_home_pose` is 497.6 / 497.4 at the statue and 456.3 / 456.0 at lambda
  0.5. Further forward r10 charges more than r9 for the same posture (paired
  on the same spawns, -8 per episode at lambda 0.625 to -33 at 0.875). The
  terms that rise with the lean (head_clearance, nosedive, posture) total
  under +10 per unit.
  The settled leg target supplies most of the restoring slope, and the
  lateral lean term is small along a symmetric lean: neither arm L reward
  option drives the drift.
- **Falls no longer stop the drift (H1a: a permissive condition, not a
  driver).** The highest held lean that at least 5 of 8 spawns survive is at
  CoP 0.89-0.94 on r10 against 0.76-0.79 on r9 (the diagnosis's spawns and
  the verifier's new ones); at least 7 of 8: 0.92 against 0.69; single
  episodes reach 1.10 (0.221 m) against 0.85 (0.201 m). Forward held leans
  fall by nosedive on both plants. On r9 the open-loop edge lies at about the
  rail (r9's toe line, 0.196 m, is pad CoP 0.80); on r10 nothing physical
  stops a forward drift before the 0.80 rail. On r9, arm L seed 60 drifted to
  pad CoP 0.86 at 5.38M and was back at 0.27 by 6.00M; on r10, seeds 52 and
  60 stayed past the rail.
- **Seeds 52 and 60 are in a worse basin (H2: consistent, mechanism not
  identified).** At 4-6M their eval returns are lower than seed 42's by 97
  and 163 per episode, no material term is higher, and 28/115 and 55/121 of
  their training rollouts at 5-6M end in a termination against 5/112 for seed
  42. Their own mean action held open-loop survives 0/8 (legs only) and 2/8
  (with head and tail): closed-loop feedback holds a posture that open-loop
  holds cannot. A held escape path back to the statue shows no barrier.
- **What started the lean is not identified.** All three r10 seeds were
  forward at the first evaluation (0.18/0.33/0.32 at 0.125M against a held
  statue's 0.11; policy std 0.218-0.219). Seed 42 made the same excursion (up
  to 0.65 at 0.62M) and came back under 0.2 by 1.50M; seed 60 passed 0.8 at
  0.88M and seed 52 at 1.75M. The only remaining evidence is a correlate:
  tail contact was 0.52-0.59 of r10's early terminations in the first 131k
  steps against 0.30-0.38 on r9 (shares, not rates). The noise mechanism of
  v1 is withdrawn (§11).
- **Reading.** The failure is plant plus training dynamics, not a reward term
  paying for the toes and not caused by arm L's two reward options (the
  settled target and the lean term; `MirrorKLPPO` was not tested, and the r9
  and r10 runs differ in harness, §6.4). On r10 the 0.80 rail is a gate
  criterion with no physical backstop between 0.8 and about 0.9, and it did
  its job: of the gate readings `gate_progress.npz` records, only the rail
  marks seeds 52 and 60.
- **Verification state.** Two verify rounds checked v1 and v2 and recomputed
  the edges and slopes quoted above. The round-3 check of v3's wording did
  not run (the spend limit, §12); v3 changed wording and scope, not these
  numbers.

## 8. Telemetry and naming

- **The r10 drift was found by hand.** Nothing in the training output or in a
  file a person could open from Drive showed seeds 52 and 60 on their toes;
  an agent decoded `gate_progress.npz` on request. #626 adds `progress.json`
  in every stage directory, the gate's two roll rails per box-soled foot,
  termination causes per evaluation and a `PostureDrift` warning line
  (over-bar fraction ≥ 0.5 on 2 evaluations in a row; advisory only),
  documented in [TRAINING_ARTIFACT_PRESENTATION.md](../TRAINING_ARTIFACT_PRESENTATION.md).
  Replayed on the r10 series, two in a row first warn on seeds 52 and 60 at
  917,504 steps and never on seed 42 through 6.03M; three in a row warn on
  seed 52 1.44M steps later (`telemetry/posture_drift_replay.json`). An
  evaluation with no full-horizon episode ends the streak, so a seed whose
  episodes all fall goes quiet exactly when worst.
- **Stand and stance (D-D35, #625).** The recipe label `stand` resolved to
  the deepest deliverable labelled `stand`, which was the recovery stage on
  three models and the stance stage on the other three, so the Colab steps
  for this screen had to tell the maintainer to type `stance` by hand. After
  the maintainer asked twice, the recovery stage's label became `recover`;
  `stand` now means stance on every model, and no digest or checkpoint moved.

## 9. Walking pilot: pre-registration (launched, not yet judged)

The fixed criteria are `pilot/criteria.txt` (sha256 `1cdd495c…`; written
2026-10-16, revised four times on 2026-10-17, all before any pilot training
step; the file's first line says "three", its Revisions list has four),
applied by `pilot/pilot_score.py`. The branch is
`claude/pilot-walk-r9` (`origin/main` `56621ebd` plus seven pilot-only
commits, head `eaeef50f`, pushed 2026-10-17 14:46, no PR). Decision D-D36 of
2026-10-17 ([BEHAVIOR_RECIPES_PLAN.md](../BEHAVIOR_RECIPES_PLAN.md) §6.2)
makes the pilot a deliberate exception to the gait plan's open GQ-3 (a) and
GQ-17 ([GAIT_QUALITY_PLAN_2026_09.md](../GAIT_QUALITY_PLAN_2026_09.md), its
dated §17).

- **What is trained.** The production T. rex locomotion stage (stock PPO, the
  production reward and hyperparameters, none of arm L's options), warm-
  started from the parent through the notebook's manual node, for 4,194,304
  steps (production 8,000,000, with production's learning-rate path), on
  seeds 42, 52 and 60.
- **The parent.** r9 arm L seed 60 at 6,291,456 steps: 40/40 clean, 6
  joint-space and 0 own-sigma noise falls; committed on the branch, zip
  sha256 `70f3a444…6f353e`, VecNormalize `0567d610…7c0fc1`. It is the
  checkpoint behind r9's K3 failure, and its run stood over the pad CoP bar
  on 39/40 episodes at 5M, one checkpoint earlier; both weaken a FALLS
  reading that points back at the stance.
- **How it is judged.** The frozen 2026-09 gait-audit probe
  ([gait_2026_09/gait_probe.py](gait_2026_09/gait_probe.py), run on today's
  API by the adapter `pilot/gait_probe_pilot.py`, no measurement line
  changed), 40 deterministic episodes on 3042-3081, floor-truth contacts, at
  2,097,152, 3,145,728 and 4,194,304; only 4,194,304 decides.
- **Episode labels, first that applies.** FALL (ends before 1,000 steps);
  HOP (≥ 0.5 touchdowns/s and simultaneous fraction ≥ 0.5); STILL (< 0.10 m/s);
  WALK, only if all six hold: alternation ≥ 0.6, lead swaps per stride ≥ 1.0,
  flight fraction ≤ 0.2, hip-pitch joint lag in [0.35, 0.65], measured stride
  frequency ≤ 5.0 Hz, and each foot's floor duty ≤ 0.80; otherwise OTHER.
- **Seed and pilot outcomes.** A seed WALKS (WALK ≥ 30 and FALL ≤ 4), HOPS
  (HOP ≥ 20), FALLS (FALL ≥ 5), is STILL (STILL ≥ 20) or MIXED; the pilot
  takes the outcome at least 2 of 3 seeds share. WALKS counts only after the
  maintainer's video review agrees (the final replay plus every
  WALK-labelled panel episode); until then the outcome is
  PENDING_VIDEO_REVIEW.
- **Why the guards.** Four verification rounds, each with its own hop
  generator run through the frozen probe, found a hop that read WALK under
  the rule of the time: a 30 ms staggered hop (→ the hip-lag band), 40 ms
  staggered hops at 9.2-10 Hz and an 8.3 Hz scissor hop (→ the stride cap),
  short-lift shuffle-hops whose 1-2-step lifts the debounce fills (→ the duty
  guard), and mixed-lift hops (→ wording; the video review is the check).
  The 0.80 duty cut came from a cost table over 24,112 hops and 297 walks
  (`pilot/duty_cut_summary.json`): 61 of 470 distinct false WALKs, all at a
  true 16-22 Hz, still read WALK at 0.80, and the real Compsognathus walker
  stays 30/30 WALK at every cut tested.
- **What each outcome would mean** (fixed before training): WALKS, today's
  walk reward starts an alternating gait from this stance by 4.2M, confirmed
  on video; HOPS, the September hop reproduces from a new stance and points at
  the locomotion reward, not the stance; FALLS, read the traces, falls while
  standing point back at the stance; STILL, too short to tell. The locomotion
  gate (`reward_and_length/v1`) is recorded, not a criterion, and is expected
  to fail on speed.

**Outcome:** launched on Colab on 2026-10-17 and not yet judged. A first
launch, at about 17:01-17:03 UTC, ran the notebook's chain loop, which began
training a stance from scratch (`01_stance`, stage 1, a 12M budget). That it
was Runtime > Run all with the manual cell at its defaults is an inference
from the notebook state (cells run in order to the chain loop, `MANUAL_NODE`
None), not confirmed. A Drive check caught it at 17:06; the maintainer
stopped the runtimes and deleted the three run folders (no `03_locomotion`
was written in them). Seeds 42, 52 and 60 were then relaunched from the
parent under the same names, `pilot_walk_s42`, `pilot_walk_s52` and
`pilot_walk_s60` (the same setup steps rerun with the old folders deleted
first, not the `_b` run ids the criteria allow for a seed restarted after a
cut-off); the new folders were created at 17:13, 17:23 and 17:24 UTC, so the
first launch ran at most about 10-12 minutes on seed 42 and 20-24 on seeds 52
and 60 (its stop times are not recorded). Each holds the run-level
`provenance.json`, `trunk_run.json` and `zero_action_baseline.json` and a
`03_locomotion` stage (created 17:17, 17:26 and 17:28 UTC) whose
`stage_config.json` reads stage 2, commit `eaeef50f`, its seed, 4,194,304
steps, `initialize_next_stage` from the parent (sha256 `70f3a444…6f353e`),
physics r9, interface r15, no resume, and hyperparameter and task digests
(`af78e500…`, `23d50a6d…`) equal to the branch's digest snapshot. The outcome
will be appended to this note as a dated note, with the scored
`pilot_score.py` output and the video review.

## 10. What did not work

- A per-step clipped CoP margin on r9 (probe 1; it saturates under the
  policy's noise).
- Re-centring `leg_home_pose` alone (probe 2; within the control's seed
  spread).
- An unbounded left-right leg quadratic (rewards falling sooner) and, at 1M,
  its bounded EMA form on top of lower noise (no measurable effect).
- Lower exploration noise (delays the edge; -2.0 does not even delay it).
- A load-gated EMA pad-CoP centring bonus (`cop`; delays, then 99/120).
- Capped sagittal leg authority, with or without a hip-roll cap (holds the
  fore-aft edge; fails on lateral roll and creep).
- Arm C's instantaneous fore-aft CoM term (met by actuator saturation, then a
  one-pad creep).
- Arm L on r9 (fore-aft creep and joint-noise falls) and on r10 (all four
  criteria).
- The r10 foot as a fix for the forward lean (it widened the survivable
  region instead).
- Reward-selected checkpoints as a safeguard (§3.4, §6.7).

## 11. Corrections and withdrawn claims

Earlier statements in the session's reports and messages, and what replaces
them. None of the superseded wording is in the repository except where noted.

| Claim (superseded or withdrawn) | Correction | Source |
|---|---|---|
| The statue's digit load is 96% at the bases | 92.4-93.6% | `run_a/report.md`, Corrections 1 |
| Under run A's hold the whole-foot CoP is 0.4 cm short of a 20.88 cm toe edge, about 4 N·m of margin | At the pitched toe-corner line (20.55-20.62 cm), 0.09 cm beyond to 1.9 cm short of it per seed; 0-8 N·m | Corrections 2 |
| Seven saturated terms sum to 3,062; energy is zero for a constant action | 3,081.7; -0.5 to -0.65 per episode | Corrections 3-4 |
| "Digit III carries the least load, against anatomy" | Overstated; the model's order follows lateral pad roll | Corrections 5 |
| Ramped ankle offsets reach 0.80 near the toppling limit (withdrawn) | The left ankle reaches 0.808 at -3.5° with 0/5 falls | Corrections 6 |
| Run A's pattern follows from linear superposition of statue slopes (withdrawn) | Its knee and ankle offsets alone fall in 10/20 | Corrections 7 |
| Probe 2's episode-level Fisher p-values (0.017, 0.005) support the settled reference; "halves the edge" (withdrawn) | 40 rollouts of one policy are not replicates: the same test between the two control seeds gives p = 0.010; the training seed is the unit | Corrections 8 |
| `leg_home_pose`'s keyframe pull is the root cause | Removing it left the edge (probe 2) | Corrections 9 |
| The leg quadratic `d_joint` passes and does not reward falling early (withdrawn) | Gains up to 157 at uniform σ 0.22, above the fall penalty of 100 (177 in the author's recheck) | `noise_rank/verify_summary.txt`, `summary.txt` |
| At 3M low noise falls about 8x and middle noise 14x as often as the control at σ 0.16; the 1M screen's "37-44/100 against 6-10/100" (withdrawn) | Unlike noise tests: the control had been tested at its own per-actuator σ (knees and ankles 0.066-0.081). Under uniform 0.16 the control itself falls 9 + 11 + 17 = 37/60; later screens used the control's vector in joint space | `cap_screens/control_s*_uniform_sigma016.json` |
| The reward does not price the lateral lean (roll diagnosis) | True of the net return only: removing the lean gains foot_flatness +10.6 to +17.9 and costs foot_load_balance -11.6 to -16.5 on 4 of 5 spawns; net spread ±32, not "-1 to +28" | `cap_screens/roll_verify_reward_terms.json` |
| The design's lean-term slope "0.0136/mm at 14 mm" | 0.0130/mm (0.0136 is the slope at about 13 mm) | `symmetry/groundwork_verify_summary.json` |
| The first symmetry research stage: per-mini-batch mirror augmentation | Replaced by the KL-form loss (§5.1) | `symmetry/mirror_ratio.json` |
| "All six runs finished with no errors, no invalid segments and no restarts" (3M interim message) | L42 and C42 fresh were re-run after the container reclaim | `symmetry/rerun_ledger.json` |
| Arm L seed 60 at 3M was "the first uncapped policy ever to pass"; "no uncapped policy had ever passed" | lownoise_asym seed 56 at 524k passed 37/40 and `cop` seed 42 at 2M passed 37/40. Arm L is the first uncapped arm to pass v2 at a screen's verdict checkpoint on 2 of 3 seeds | `cap_screens/screen_1m_analysis.txt`, `screen_3m_cop_authority_analysis.txt` |
| The r10 certificate "judged robust_best_model.zip at 12,001,280 steps" ("12M: PASS/FAIL") | It judged reward-selected checkpoints (about 262k steps on seed 52, 131,072 on seed 60, about 4.3M on seed 42); 12,001,280 is the run total | §6.7 |
| r10 seed 60's falls per 1M bin "9, 10, 13, 9, 9, 2" (sum 52) | 10, 12, 20, 12, 13, 2 (sum 69) | `r10_screen/report_only_summary.json` |
| Diagnosis v1: r10's digits keep a forward lean standing out to CoP 1.06-1.10 against about 0.85 on r9 | 1.06 was a level with 2 of 8 survivors; use 0.89-0.94 (at least 5 of 8 survive) against 0.76-0.79, and 1.10 against 0.85 for single episodes. The v1 figure survives in the public #623 result comment | `diagnosis/findings_v3.txt` |
| Diagnosis v1: reward slopes -192 (r10) and -196 (r9) per unit CoP; `leg_home_pose` "penalises less on r10" | Pooling artefacts; -109 / -109 on identical postures, and `leg_home_pose` is the same per posture on both plants | `diagnosis/findings_v3.txt` |
| Diagnosis v1 (H1): under exploration noise the r10 statue falls backward and r9's forward, so a forward lean pays on r10 (withdrawn) | Not on 20 new spawns: at σ 0.10 their paired difference was -8 [-237, +229]; pooled over all 30 pairs it is +149 [-67, +381], which includes zero; at σ 0.15 and 0.21 the lean costs on r10 too | `diagnosis/findings_v3.txt` §4 |
| Diagnosis v2: the walking pilot starts at the stance's initial noise (std 0.218) | SB3's load restores the parent's learned std (about 0.09); the locomotion warm-up's effect on it is unmeasured | `diagnosis/findings_v3.txt` §7 |
| PostureDrift's default: "a streak of three would have missed seed 52" | True only of a snapshot that ended one evaluation short; on the full series three warn 1.44M steps after two. Only the corrected wording landed (#626) | `telemetry/posture_drift_replay.json` |
| Each WALK-rule revision's own claim ("a staggered hop cannot pass"; "staggered or scissor hops above 5 Hz read OTHER") | Each falsified by the next round; the final text states limits as tested | `pilot/criteria.txt`, Revisions |
| The foot proposal's comparison table: the r9 foot's print W/L 0.43 at the contact line and 0.65 at the outer margins; the floor-moment gain +4.9% | 0.39 and 0.61; +4.6% from unrounded values (as the #623 body states) | the proposal's second review (not copied), #623 |

## 12. Method lessons

- **Pre-register, then keep the fails.** Every screen's criteria were written
  before it trained, with dated clarifications only (r10: criteria 13:59,
  tie-break 16:30, first run 17:47). Fails stayed fails: the 6M authority
  arm's concave E2 series and the hip-roll arm's H2 miss by one episode were
  not re-judged with a better-fitting fit or relaxed bars, and no run was
  changed mid-screen (arm C saturating at 3M; r10 seeds 52 and 60 on their
  toes at 2.3M). Rule amendments made after seeing data were disclosed with
  both picks (the noise check's E5; the fore-aft term's selection rule).
- **Keep pre-registration light.** After the maintainer's gut check of
  2026-10-16 the four-version symmetry design with frozen hashes and a
  nine-outcome ladder gave way to four fixed criteria, one comparison and an
  independent verification, with the same rigour in the verdict.
- **Dry-run the verdict scripts on stored data.** Running the frozen symmetry
  verdict on old screens before training caught a class statistic (an
  unsigned median pad difference) that would never have fired on the policy
  it was built for; the final design replaced it with the signed pad bias
  that `symmetry/frozen_params.json` records (the design's final-revision
  record; session scratch only).
- **No significant regression is not non-inferiority.** The foot proposal's
  first merge rule (one-sided Fisher p ≥ 0.05 against the r9 arm) would have
  passed 16/60 joint-noise falls against 8/60 (p = 0.054), the very counts
  the r10 screen later got (K3); a review caught it (session record), and the
  r10 screen used fixed bars instead.
- **Verifiers re-implement.** Independent rollout, decode and statistics code
  agreed on 30/30 re-judged cells (1M screen), 120 and 537 verdict quantities
  (symmetry), and every r10 cell. Verifiers err too: the 1M screen's
  verifier introduced the unmatched noise comparison that the next screen's
  verifier refuted.
- **Matched noise means the same noise vector for every policy:** the
  production control's per-actuator sigma, applied in joint space.
- **The training seed is the unit of replication.** Compare per-seed counts
  (exact rank-sum), not 40 rollouts of one policy.
- **Test reward terms for a falling-sooner incentive per trajectory**, and
  prefer instantaneous terms: PPO credits a 100-step EMA term at only
  0.13-0.33 of an instantaneous one (`symmetry/ema_credit.json`).
- **Report per-term returns.** A flat net return hid foot_flatness against
  foot_load_balance (±32).
- **Rank candidates under the policy's own noise as well as
  deterministically**; offline rankings on policies trained without a term
  did not predict training (arm C).
- **Judge several checkpoints.** Arm L seed 60 went 0/8/39/0 CoP failures over
  3-6M; on two r10 seeds the reward-selected "best" checkpoints were 131k- and
  262k-step policies.
- **Hold physics changes as drafts until a training screen.** Zero-shot,
  r9-trained policies looked better on r10; policies trained on r10 found a
  toe-standing basin.
- **Correct before publishing.** The v1 diagnosis figure "1.06-1.10" reached
  the public #623 comment after v3 existed; "first uncapped policy ever to
  pass" and "no restarts" were contradicted by files already on disk.
- **Report-only first, then prove it inert.** Each telemetry change showed the
  digest snapshot unchanged; #626 also showed a short training run identical
  with and without it.
- **A regression test can skip silently in CI.** #621's test skipped without
  SB3 and was named in no SB3 job step until review caught it;
  `test_ci_tool_pins.py` now requires every such file to be named there
  (CHANGELOG, #621).
- **Telemetry gaps decide what gets noticed.** Run A's left-foot CoP was not
  recorded (→ #620); a resume dropped gate rows (→ #621); the r10 drift was
  visible only to a hand decode of a Drive file (→ #626).
- **Names mislead.** `stand` meaning different chains on different models
  (→ D-D35), "N/40" read as failures, "12M: PASS" on an early checkpoint,
  "the stance gate" for a single-heading panel: each was caught by a
  reviewer or the maintainer, not by the author.
- **Infrastructure.** Two container events killed four training
  seed-segments; the predeclared one-re-run rule kept the screen valid.
  Runs longer than a couple of hours went to Colab afterwards. When the
  spend limit stopped the agents, the reporting change's checks were re-run
  directly before it opened, and the unverified pieces are labelled (§7).
- **Keep caches out of the checkouts.** Agents wrote bytecode or a ruff cache
  into the main checkout at least twice, and once bytecode into the frozen
  symmetry screen's harness; all were git-ignored or removed and the frozen
  sums still passed (session record). Set `PYTHONDONTWRITEBYTECODE=1` and a
  scratch ruff cache directory.
- **Check a Colab launch on Drive at once.** The pilot's first launch ran the
  chain loop with the manual cell at its defaults (by the notebook state,
  Runtime > Run all; not confirmed), so it started training a stance from
  scratch; reading the new run folder (a `01_stance` stage with a 12M budget,
  no `03_locomotion`) caught it a few minutes later, at 17:06 (§9).

## 13. Open questions and next steps

Decisions below are the maintainer's; none is taken here.

- **The walking pilot's outcome** (§9), and what follows from it: on WALKS, a
  certifiable walker still needs a certified stance under a committed stance
  config (decisions on #624's options and on #623); on HOPS, the gait plan's
  section 5 (`gait-r1`) is the indicated route.
- **What starts the early forward push on r10**, and what holds seeds 52 and
  60 in their basin. Both need the Drive checkpoints (§14), and the first
  needs training-side runs that the stance-only policy rules out unless
  walking points back to stance.
- **Whether a revised foot keeps r10's gains without making a toe-standing
  lean survivable past the rail** (for example leg-scale digits, where the
  push limit falls to 49.8 N, or softer MTP springs): a new physics decision
  and a new screen. #623's open questions (`MEASUREMENT_VERSION` v3,
  `max_settle_touchdowns = 2`, the recovery push re-freeze at 186.8 N) are
  moot while it is parked.
- **What the stance stage should certify beyond the statue.** The statue and
  50k-step checkpoints pass v2; the screens' noise tests (H4, K3) partly
  measure what trained policies add, but no committed gate does.
- **Pad or whole-foot CoP for the T. rex bar**, a lateral CoP bar, sole rails
  for the Compsognathus robot, and which hinge counts as the ankle on the
  other box feet (#622 left it open).
- **The hip-roll reset noise** (0.05 rad) sets 35% (authority arm) to 62%
  (hip-roll arm) of the failing lateral roll (roll diagnosis option c): a
  task-design decision.
- **Dibothrosuchus:** its statue sits 0.31° of ankle offset from a 0.80 CoP
  bar on the heel side while its trackways are digitigrade; to settle before
  any quadruped CoP rail.
- **Symmetry follow-ups not taken up:** kappa x3, the E2 fallback, the
  fore-aft term's recalibration, the epochs confound, and whether the lean
  term and the re-centring are needed.
- **The creep test.** A linear 12M projection fails concave series (the 6M
  authority arm); is it the right test for a future screen?
- **Tooling.** The frozen 2026-09 gait probe no longer imports on `main`
  (D-D33 removed `SPECIES_FACTORIES`); the pilot runs it through an adapter.
  The notebook's end-of-stage certificate judges the reward-selected
  `robust_best_model`, which its output does not say. A same-stage resume in
  the first 2M locomotion steps skips the forward-weight ramp (measured:
  `forward_vel_weight` 2.0 and `ent_coef` 0.00497 after a resume at 16,384;
  KNOWN_ISSUES has the entry, read from code).
- **The public #623 comment** quotes the superseded "CoP 1.06-1.10, against
  about 0.85 on r9" (§11).

## 14. Where the evidence is

- **This repository:** [stance_2026_10/](stance_2026_10/README.md), with
  `SOURCES.txt`, which maps every copy to its source, a README that says how
  to re-check each verdict, and `SHA256SUMS`. The walking pilot's decision,
  D-D36, is in [BEHAVIOR_RECIPES_PLAN.md](../BEHAVIOR_RECIPES_PLAN.md) §6.2,
  with the gait plan's dated §17.
- **Pull requests and branches** (none merged unless stated):
  - #623 ([draft](https://github.com/kuds/mesozoic-labs/pull/623), parked):
    branch `claude/trex-foot-r10`, head `0ee0917d`; carries
    `docs/investigations/TREX_FOOT_R10_2026_10.md` and D-D34; the screen
    result is its [comment of 2026-10-17](https://github.com/kuds/mesozoic-labs/pull/623#issuecomment-6234386057).
  - `claude/screen-trex-r10-arm-l` at `026eb833`: the r10 screen's training
    branch (#623's foot plus arm L on).
  - `claude/pilot-walk-r9` at `eaeef50f`: the walking pilot; its
    `pilot_checkpoints/r9_arm_l_s60_6291456/` holds the parent and
    `SOURCE.txt`.
  - #620, #621, #622, #624, #625 and #626 are on `main`; their landing
    records are in the status table of
    [CONSOLIDATION_PLAN_2026_09.md](../CONSOLIDATION_PLAN_2026_09.md)
    (#626's added with this note).
- **Google Drive:**
  - Run A: `MyDrive/mesozoic-labs/logs/trex/ppo/20261011_165957/` (the
    layout before D-D33's binomial species ids and embodiment axis); its
    gate report is copied (`run_a/`).
  - The r10 screen, under `MyDrive/mesozoic-labs/logs/tyrannosaurus_rex/anatomical/ppo/`:
    `r10_armL_s42/`, `r10_armL_s52/`, `r10_armL_s60/`
    (`01_stance/models/` with the judged 4M, 5M and 6M checkpoints and their
    VecNormalize files, `gate_progress.npz`, `evaluations.npz`). The judged
    files' sha256 are in `r10_screen/fetch_record.json`; the best stance by the fixed rule is
    `r10_armL_s42/01_stance/models/stage1_6291456_steps.zip`, sha256
    `7c3c1df77be4d041fe51e48fec45e70ded2fde34e70af6b111038444250cb51d`
    (VecNormalize `6e22ae9aef095062a23ff358cac5ed269d74084012fd58e595797ef094ed40c4`).
  - The walking pilot writes `pilot_walk_s{42,52,60}/03_locomotion/` under
    the same directory: the folders of the relaunch of 2026-10-17 (created
    17:13, 17:23 and 17:24 UTC), each also holding the run-level
    `provenance.json`, `trunk_run.json` and `zero_action_baseline.json`. The
    first launch's folders, which held a stance trained from scratch, were
    deleted before the relaunch (§9).
- **Not copied** (session scratch only): the r9 screens' checkpoints, rollouts
  and traces (about 1.7 GB; the r9 arm L seed 60 6M pair survives on the pilot
  branch, sha256
  `70f3a4447794a01036c27bca5e0377a17db96d0e3d7bfb2d179e9ce94e6f353e` and
  `0567d6104a0795c0201ff8db4b1045e907bf3100ca9ae60fa4028c4f907c0fc1`),
  the per-episode judge files, the probes' raw episodes and the session's
  workflow journals.

## 15. Status 2026-10-18 (appended): pilot 1's outcome, the offline follow-ups and pilot 2

*Appended 2026-10-18; §1–§14 above are unchanged, so their "not yet judged"
is the wording of 2026-10-17.* This section records walking pilot 1's
outcome, the offline work of 2026-10-17 that followed the note (the stage's
own certificate rolled on the r9 arm L checkpoints, an offline battery on
them, a foot and ankle audit, a literature report and a recommendation on
what to log next) and the maintainer's decision of 2026-10-17 to run a
second pilot. Pilot 1 is the only part that trained. Everything else is
offline and report only: the certificates were rolled on uncertified
checkpoints that the judge would refuse, and nothing here certifies a run,
writes a result bundle or promotes anything. The evidence is in seven
folders added to [stance_2026_10/](stance_2026_10/README.md) on 2026-10-18
(its README's "Added on 2026-10-18" says what each holds and how to re-check
it); paths in backticks below are relative to that directory. The 28
judgements among them (19 certificates and pilot 1's 9 panels) are rows of
the repository's ledger,
[`results/offline_judgements.csv`](../../results/README.md#offline-judgements-offline_judgementscsv),
each pointing at its evidence file.

### 15.1 Walking pilot 1: STILL on all three seeds (judged 2026-10-17, final)

- **The runs.** All three seeds finished 4,194,304 steps on L4 runtimes
  (3 h 36 min to 3 h 41 min), with no collapse and no early stop. The judge
  bound each checkpoint to the branch's tree `eaeef50f`, the parent's sha256
  `70f3a444…`, the budget and the plant, and accepted all nine
  (`pilot1_judging/run_log.txt`;
  `pilot1_judging/OUT/s<seed>/<steps>/binding.json`).
  The small files fetched from Drive were transcribed twice and matched (16
  files), and seed 42's 2M VecNormalize file, repaired by hand at the 2M
  pipeline check, matched its final fetch (`run_log.txt`).
- **The outcome**, by the criteria of §9 (`pilot/pilot_score.py --seeds
  42,52,60`; `pilot1_judging/OUT/score.json`, sha256 `86d7350e…` before the
  copy's final newline; printout `pilot1_judging/judge_logs/score.log.txt`):
  at 4,194,304 steps every seed is STILL on 40 of its 40 panel episodes, with
  no WALK, HOP or FALL episode; median speed 0.001-0.002 m/s, no touchdowns,
  flight 0, foot duty 1.00 on each foot. **PILOT OUTCOME: STILL (3 of 3
  seeds).** It is final; no video review was needed, since only WALKS needs
  one. On the trajectory, which decides nothing, every cell at 2,097,152 and
  3,145,728 is STILL 40 except seed 42 at 2,097,152 (STILL 39, FALL 1: one
  episode sliding at 0.108 m/s, ended at step 837; `run_log.txt`).
- **What STILL means, as fixed before training** (§9; `pilot/criteria.txt`,
  "What each outcome means"): the policy is not moving by 4.2M, and the pilot
  was too short to say anything about the gait. By the same text only FALLS
  could have pointed back at the stance; no seed fell at 4,194,304.
- **Report only** (`run_log.txt`, and the runs' own `gate_progress.npz`
  where stated; none of it is a pilot criterion):
  - The in-training evaluation reward was about 1,097-1,100 at the 2M, 3M
    and 4.2M evaluations, the parent's standing reward under the walk task
    (the run log's "flat at ~1,097-1,100 throughout"). Over all 96
    evaluations (`gate_progress.npz`) it was 1,093-1,101 wherever all ten
    episodes ran the full horizon; 10 of the 96 dipped to 993-1,080, each
    with one of its ten episodes ending early (`full_horizon_fraction`
    0.9; seed 60 7 times, seed 52 twice, seed 42 once).
  - The forward-velocity term was 0.5-3.2 per episode at the 2M, 3M and
    4.2M evaluations, against the September T. rex walkers' 39-54 at 2.1M
    and 148-234 at 4.2M. Over all 51 evaluations from 2M to 4.2M it was
    0.1-7.5, 12 of them outside 0.5-3.2 (`gate_progress.npz`); the run
    log's "0.5-3.2 at 2-4.2M (max 7.5)" is the range at those three
    evaluations.
  - The `gate_progress.npz` figures were read on 2026-10-18 from each
    run's `03_locomotion/gate_progress.npz` on Drive (32 evaluations, the
    last at 4,194,304; folder ids in `run_log.txt`), transcribed twice for
    the final judging and byte-identical (sha256 `86896545…`, `b0317e62…`
    and `29858fd4…` for seeds 42, 52 and 60); they are not copied into
    the evidence folder.
  - The policy's action std (exp(log_std), mean over the 15 actuators, read
    from the checkpoints by `pilot1_judging/tools/stdcheck.py`) was 0.095 in
    the parent (min 0.018, max 0.164, as `pilot2/criteria.txt` quotes it) and
    0.108-0.111 at 4,194,304 (min 0.022, max 0.196): exploration barely
    widened. §11's "Diagnosis v2" row left the locomotion training's effect
    on the parent's learned std (its "about 0.09") unmeasured; these two
    readings, the parent and the 4,194,304 checkpoints, are the only ones
    taken.
  - The notebook's locomotion gate (`reward_and_length/v1`) read FAIL on
    speed (0.00 against 1.00 m/s), as expected; recorded, never enforced.
- **Hypotheses, not findings.** (a) With its noise at about 0.1, under half
  the stance stage's starting 0.2231, the training rollouts barely left the
  parent's standing actions, so they rarely produced the displacement the
  forward term pays, and the reward had little to pull toward stepping (the
  reason given for pilot 2, §15.7). (b) Today's walk reward does not pay
  stepping from this stance within 4.2M steps at any noise. Pilot 1 cannot
  tell them apart, and neither was tested.

### 15.2 The stage's own certificate, rolled offline (2026-10-17, report only)

The stage's certificate (`stance_quality_heading/v1`, D-D32) had never judged
an arm L checkpoint. `offline_certificates/scripts/run_cert.py` called
`build_stance_heading_report`, the function the notebook's end-of-stage
certificate calls, on `main` `2e734e29` (physics r9) with `stance.toml` as
committed (options off) or with arm L's three env options (armL), on the r9
arm L symmetry-screen checkpoints of §5
(`offline_certificates/offline_certificates.txt`; the table is in the
evidence README):

- At 6,291,456 steps on 3042-3081: seed 60, the pilots' parent, **PASS**
  40/40 seeds clean at all six headings (LCB 0.9278); seed 52 **PASS** 38/40
  (0.8508), failing pad CoP on seeds 3050 (0.997) and 3063 (0.823); seed 42
  **FAIL** 29/40 (0.5861), on pad CoP, sole tilt and corner lift. Options off
  and armL give the same verdicts and identical floor metrics: the options
  change only the reward (`offline_certificates/cert_summary.json`).
- The reward-selected `robust_best` checkpoints (seed 60 at 524,288, seed 42
  at 655,360 and seed 52 at 917,504 steps) **PASS** 40/40, 40/40 and 39/40.
- The fresh audit panel 7042-7081 passes both 6M checkpoints it was run on
  (seed 60 39/40, seed 52 38/40), but both fall on seed 7047, where the
  statue stands, so each sits on the `max_hop_or_fall_episodes` rail of 1.
- **Margin.** The forward pad CoP is the bar that binds: seed 60's worst
  full-horizon pad CoP is 0.780 against the 0.80 bar, the statue's 0.475
  (`offline_certificates/cert_margins.txt`).
- **None of it is gate evidence.** The runs were trained in the session's
  scratch harness (`MirrorKLPPO` with scratch `experiment_*` keys; not on
  Drive) and recorded task `a2af3c5c…`; these certificates roll task
  `bde44225…` (off) or `cd963518…` (armL), so the judge would refuse them on
  task, and an audit panel is refused as gate evidence by design
  (`offline_certificates/evidence_table.txt` §E). Each verdict was re-derived
  from its per-episode panel by the library's gate (all 11 agree,
  `offline_certificates/rederive_from_csv.json`).
- **How close** (`next_signals/recommendation.md` §1): close in behaviour,
  not in certification. The stage needs two certified seeds under one
  committed config and has none: the passing runs used a recipe
  `stance.toml` does not enable, their `robust_best` pairs were selected under
  the scratch reward, and the only production-config r9 run, run A, was
  clean on 1/40 (§3).

### 15.3 The offline battery (2026-10-17, report only)

`battery/battery_report.md`, on the same checkpoints, `main` `2e734e29`,
options off, no training:

- **Fresh seeds.** Eight more certificates on 5042-5081 and 9042-9081 all
  PASS: seed 60 6M 40/40 and 39/40, seed 52 6M 37/40 (LCB 0.8174, the
  minimum passing count; 36/40 would give 0.7856) and 39/40, both
  `robust_best` checkpoints 40/40 on both (`battery/fresh_seeds/`). No policy
  episode fell, and every unclean fresh seed failed the pad CoP bar. Pooled
  over the four panels, which is not a gate statistic, seed 60 6M is clean on
  158/160 (exact 95% interval 0.9556-0.9985) and seed 52 6M on 152/160
  (0.9039-0.9782)
  (`battery/fresh_seeds/fresh_seed_summary.txt`); at seed 52's pooled rate,
  and assuming independent seeds, about one fresh 40-seed panel in seven would
  fail (`battery/report/panel_pass_prob.json`, an illustration). Seed 42 6M
  had no fresh panel.
- **The shared fall on 7047** (`battery/shared_fall/`): the reset leaves the
  animal tipped forward, its most toe-down and forward start of the 40 audit
  seeds. The statue survives by folding (its hips sag and its heels come
  down); both policies hold the hips at or above home, pin the pad CoP on the
  toe edge and topple forward (seed 60 by head contact at step 173, seed 52 by
  nosedive at step 212), with no actuator saturated and nothing slipping.
- **Push and release** (`battery/push/`): `main`'s impulse probe as shipped
  cannot tell the policies apart, as its grid starts above every envelope. On
  a 16-seed copy with a finer grid and fore-aft rows, the weak direction
  follows where each policy parks its pad CoP: seed 60 6M, toe-loaded (+0.31
  and +0.36), survives forward pushes of 0.05 m/s on all 16 seeds (0.20 on at
  least 8), against the statue's 0.15 (0.25), and the heel-loaded seed 52 is
  weak backward. No policy beats the statue in every direction.
- **Own noise** (`battery/own_sigma/`): on the heading-0 panel with actions
  sampled from each policy's own Gaussian, no policy falls and the 6M verdicts
  hold, but every `robust_best` checkpoint is clean on 0 of 40 (settle force
  over 1.5 body weights on every seed, and pad CoP), against 39-40 when run
  deterministically; its own leg noise (0.114-0.152) is about three times the
  6M checkpoints' (0.036-0.045). The reward-selected handoff is clean only
  when run deterministically.
- **The report's verdict:** seed 60 6M is a real but thin pass, and seed 52 6M
  is on the rail.

### 15.4 What the fore-aft bar does not see

Measured by the battery and the audit, and changed by neither; a gate change
needs the maintainer's approval.

- **Pad only.** The bar reads the pad's own CoP, not the whole foot's or the
  COM's. With 16-28% of the load on the digits at the MTP joints, the pad CoP
  reads 0.10-0.25 pad half-lengths ahead of the COM projection (seed 60 +0.27
  against +0.11; the noise-free statue +0.31 against +0.06)
  (`audit/foot_ankle_audit.md`, "The CoP statistic in the anatomical frame").
- **Unsigned.** The bar bounds |CoP|, not the side it sits on. Seed 60 6M
  passes it (median 0.36, worst 0.78) and has the smallest forward push margin
  of any role tested (§15.3; `battery/battery_report.md` §4). A track-like
  stance with its load over the MTP joints reads 0.86 and would fail, and seed
  52 failed reset seed 3050 at the heel edge (-1.00) (the audit). Seed 52 sat
  heel-ward on both pads in 30 of its 48 in-training evaluations, seeds 42 and
  60 toe-ward on 46 and 47
  (`next_signals/diagnose/signed_pad_cop_r9armL.txt`).
- The battery's first next step is to report the signed pad CoP beside the
  unsigned one, report only; the audit adds that a signed whole-foot
  statistic would close both gaps. Neither is decided.

### 15.5 The foot, ankle and posture audit (2026-10-17, measurement only)

`audit/foot_ankle_audit.md`, on `main` `2e734e29` (physics r9), with the
published values restated in the model's frame (`audit/literature/out/`):

- **Nothing measured in the anatomy pushes the quiet stance forward.** In the
  statue and both 6M policies the COM projects 2.4-6.8 cm ahead of MTP III,
  inside the proximal-phalanx zone where tyrannosaur tracks are deepest, and
  none of the three stances leans forward (each subvertical and slightly
  flexed, every joint within 5° of home;
  `audit/posture/out/posture_table.txt`).
  The drift state itself (seed 60 at 5M) was not rolled.
- **Matches:** footprint over hip height (0.260, published 0.216-0.270),
  digit lengths, the load centre, the COM ahead of the hip (41.9-45.6% of
  femur length against 30-45%, the top of the band once settled), the knee's
  range, and a knee near the ground-force line.
- **Departures:** the COM sits 4-18 cm too high relative to the hip in every
  pose; the thigh is light, the forelimbs heavy, and the shank and metatarsus
  proportions juvenile (metatarsus over femur 0.613, tibia over femur 1.000,
  against adults' 0.52-0.60 and 0.90-0.93); the digit spread is 3-5× too
  narrow and in reversed order (roll, not fore-aft); the rear contact is a
  midline pad, not a lateral digit IV pad; the ankle extends 11.2° past the
  only published T. rex model range (Hutchinson et al. 2005, 90-180°
  included). The tail cannot lift off its stop, so it has no static fore-aft
  authority, and the neck and head move the COM rearward only.
- **Two features let the stance read, or go, further forward:** the pad-only
  reading of §15.4, and the rigid pad front edge, 4.6-6.0 cm beyond the
  published load zone; the 0.80 bar sits 3-5 cm ahead of that zone (the
  audit's inference).
- **Seven candidate physics changes, none decided, each needing the
  maintainer's approval:** lower the COM into the published band; cap ankle
  extension at 180°; pull the rigid tipping edge back to about 0.07-0.08 m
  ahead of MTP III; a lateral digit IV rear pad; a wider digit spread; a tail
  that can hold level; adult limb proportions. The audit supports neither
  changing the neck or head nor extending rigid support forward.

### 15.6 The literature report and what to log next (2026-10-17)

`literature/trex_foot_and_standing_balance.md` ("Longer toes cannot cure a
forward lean") and its four research notes. Where its model figures differ
from the audit's, the audit's "Corrections to the literature report"
supersede them.

- **The published record:** a standing T. rex, like the birds, carried its
  weight on the proximal parts of its digits near the toe joints, with its
  COM about 30-45% of a femur length ahead of the hip; in human and robot
  balance, robustness comes from the distance between the centre of pressure
  and the tipping edge, not from foot length.
- **Its reading:** the drift onto the toes is a control problem, not an
  anatomy problem. No reward term pays for fore-aft placement, the policy's
  foot channels report each foot's total load and not where it sits, a
  left-right mirror loss cannot act on a symmetric fore-aft lean, and
  extending the support forward (r10) moved the edge the policy drifted to.
  In footprint terms the 0.80 bar already accepts stances forward of the human
  quiet-stance analogue and far forward of the bird's (load under the toe
  joint); a target nearer the pad centre has anatomical support, but moving
  the gate needs approval.
- **Its ranked ideas:** first, needing no approval, log the stability margin
  and the pad and digit load split, and select checkpoints by margin and
  action-noise survival rather than reward, with about eight seeds per arm;
  then, with approval, a fore-aft load signal for the critic or the actor, a
  floor on exploration noise or training under injected action noise, and,
  only if walking points back to stance, a CoP- or ZMP-centring term.
- **Its limits:** no measured T. rex ankle or toe range of motion (the audit
  adds Hutchinson et al. 2005's model range), no standing pressure data for
  any large theropod or ratite, and no RL ablation of a CoP term in standing.
  That shrinking exploration noise lets a stance drift toward the edge is, in
  its words, a hypothesis to test. It cites a Cassie study (Li et al. 2024),
  in which stepping rescued leans that stance-only policies fell from, as the
  clearest reason to expect the walking pilot to say something about stance.
- **What to log and sense next** (`next_signals/recommendation.md`): sixteen
  numbered items, from report-only telemetry (a per-episode evaluation
  ledger, whole-body balance, a signed CoP trend) through evaluation and
  selection (keeping the onset checkpoints, the offline battery of §15.3) to
  policy-interface changes (pad and digit touch sensors, a privileged critic),
  each with the approval it needs. None could be added to pilot 1, whose
  callbacks are assembled before training and whose branch predates #626.
  The battery of §15.3 carries out item 9's fresh panels, seed-7047 replay,
  push-and-release and own-sigma parts (not its "what the policy senses on
  its own states"); no other item is taken here.

### 15.7 Walking pilot 2: pre-registration (decided 2026-10-17; not yet trained, no result)

After pilot 1 was judged, the maintainer chose "Second pilot, wider
exploration" on 2026-10-17: pilot 1 again with one change. D-D36's
amendment of 2026-10-18 records it
([BEHAVIOR_RECIPES_PLAN.md](../BEHAVIOR_RECIPES_PLAN.md) §6.2; the gait
plan's §17 note of the same date). The fixed criteria are
`pilot2/criteria.txt` (written 2026-10-17 and amended 2026-10-18, both before
any pilot-2 training step), and `pilot2/colab_steps.txt` is how the runs are
launched.

- **The one change.** At the warm start only (`initialize_next_stage`; never
  on a resume, a fresh start or a root), every one of the 15 actuators'
  `log_std` is set to -1.5, std 0.2231, the T. rex stance stage's own starting
  noise (`log_std_init = -1.5`, D-D27), after the parent's parameters load and
  before the first update, with the optimizer's moments kept. The maintainer
  was told 0.22; the pilot uses -1.5 exactly, so its noise is the stance's own
  starting noise and not a new number. A pilot-only key on the branch,
  `[curriculum] warm_start_log_std = -1.5` in the locomotion stage, sets it;
  on the branch it moves that stage's hyperparameter and stage-config-view
  digests (PPO hyperparameters `ec9925f9…`, pilot 1's `af78e500…`), and the
  task and gate digests are unchanged (`23d50a6d…`, `02602cb0…`). It leaves
  the policy's mean action, the value network and the normalisation statistics
  as pilot 1 had them, so the deterministic panel would read the parent's
  STILL before any update; the noise acts only through what training learns.
- **Everything else is pilot 1's:** the parent (r9 arm L seed 60 at 6,291,456
  steps, sha256 `70f3a444…`), the production locomotion stage and its
  4,194,304-step schedule, seeds 42, 52 and 60, the frozen probe, the episode
  labels and the WALK rule's guards, the seed and pilot outcomes, the video
  review, and pilot 1's judging tools, unchanged (their sha256 are in the
  criteria). Three rules are added because they follow from the change: the
  judging reads each run's `stage_config.json`, and a run that does not record
  the reset (`curriculum.warm_start_log_std` and `run.warm_start_log_std` both
  -1.5) is not scored and counts as missing; and a resume never applies the
  reset.
- **Where it runs:** the branch `claude/pilot-walk-r9-b` at `ba464855`
  (pilot 1's `eaeef50f` plus three pilot-only commits: the key with the
  `train_base` code that applies it, its tests and the regenerated digest
  snapshot; the printed "Warm start" line; and a note in the parent's
  `SOURCE.txt`), pushed with no pull request and never merged; run IDs
  `pilot2_walk_s42`, `pilot2_walk_s52` and `pilot2_walk_s60`; pilot 1's
  folders are never reused. The pre-launch checks on that branch (a
  `QUICK_TEST` notebook smoke and the judge run on it on the session's
  machine, the key's tests and the shared and T. rex test suites) are listed
  in the criteria; none is a pilot-2 training step.
- **What each outcome would mean** (fixed before training; a sequential
  comparison with pilot 1, not a randomized one, and GPU training is not
  bit-reproducible): WALKS, the reset was sufficient from this stance
  (not shown necessary); HOPS, the walk reward pays the hop once exploration
  widens, which points at the locomotion reward (the gait plan's §5), not the
  stance; FALLS, the traces say whether the wider noise broke the stance or
  stepping started and failed; STILL again, either 0.22 is not enough or the
  reward does not pay stepping by 4.2M, which this pilot cannot tell apart
  (the report-only std and forward-term trajectories say which is likelier).
- **Status:** not yet trained; no result. Like pilot 1 it is a deliberate
  exception to the gait plan's GQ-3 (a) and GQ-17, under D-D36: nothing it
  trains is certified, published or reused, and no result bundle is written.
  Its outcome will be appended as a dated note.

### 15.8 What changes in §13 and §14

- **§13, the walking pilot's outcome:** pilot 1 read STILL, so neither its
  WALKS nor its HOPS branch arises; pilot 2 (§15.7) is pre-registered and
  not yet trained. The other open questions of §13 stand. The follow-ups
  sharpen its "Pad or whole-foot CoP for the T. rex bar" with a signed
  statistic (§15.4), and add two: whether the reward-selected `robust_best`
  should stay the handoff when it is clean only without its own noise
  (§15.3), and which, if any, of the audit's seven physics changes to take
  (§15.5).
- **§13, the public #623 comment:** it already opens with a correction
  (updated 2026-10-17 18:54 UTC) giving 0.89-0.94 against 0.76-0.79 and 1.10
  against 0.85, and marks the in-text 1.06-1.10 as superseded (a read-only
  check on 2026-10-17, `next_signals/diagnose/notes_numbers.txt` item 2). No
  further edit is needed.
- **§14, this repository:** the seven folders above
  (`pilot1_judging/`, `pilot2/`, `offline_certificates/`, `battery/`,
  `audit/`, `literature/`, `next_signals/`), with their sources and sha256 in
  `SOURCES.txt` and `SHA256SUMS`, and the 28 ledger rows.
- **§14, branches:** `claude/pilot-walk-r9-b` at `ba464855` (pilot 2), beside
  `claude/pilot-walk-r9` at `eaeef50f` (pilot 1); neither has a pull request.
- **§14, Google Drive:** pilot 1's checkpoints are in
  `pilot_walk_s{42,52,60}/03_locomotion/models/` under
  `MyDrive/mesozoic-labs/logs/tyrannosaurus_rex/anatomical/ppo/` (folder ids
  in `pilot1_judging/run_log.txt`, sha256 in each `binding.json`); pilot 2's
  runs are to write `pilot2_walk_s{42,52,60}/` beside them.
- **§14, session scratch only:** the per-episode certificate files (their
  sha256 and sizes at the end of `SOURCES.txt`), the scratch-harness
  checkpoints the certificates rolled (sha256 in each `run_summary.json`;
  only the seed 60 6M pair survives elsewhere, on both pilot branches),
  pilot 1's traces and plots, the battery's per-episode panels and traces,
  and the audit's downloaded papers and posture rolls.

## 16. Status 2026-10-18 (appended): the offline diagnosis and three full stance screen runs

*Appended 2026-10-18; §1–§15 above are unchanged.* After pilot 1 read
STILL (§15.1), the maintainer asked on 2026-10-18 for three T. rex stance
runs with "different reward settings, slightly different mechanics, or a
refinement". The offline diagnosis of 2026-10-18 (rolls of existing
checkpoints, of the zero-action statue and of scripted held poses; nothing
trained) designed them, and the maintainer took decision D-D37
([BEHAVIOR_RECIPES_PLAN.md](../BEHAVIOR_RECIPES_PLAN.md) §6.2): three full
12M stance screen runs on `main`'s physics r9 instead of walking pilot 2,
which is parked. This section records what the diagnosis found and the three
runs. Their evidence is the folder `stance3/` added to
[stance_2026_10/](stance_2026_10/README.md) on 2026-10-18:
`recommendation.md` (the diagnosis' summary and the run design, as the
maintainer received them), `criteria.txt` (byte-identical, sha256
`1ce910dc…`), `colab_steps.txt`, and frozen copies of the judge, the scorer
and the scorer's tests. The diagnosis' tables and scripts stay in the
session's scratch under `scratchpad/foot/tonight/` (the `T/` of
`recommendation.md`); each number below names the file it comes from, paths
relative to that directory, and "checked" marks a number re-read from that
file on 2026-10-18 for this section.

### 16.1 What the offline diagnosis found (2026-10-18, no training)

Figures are on the r9 plant at heading 0, per 1,000-step episode or per step
of the 800-step window after step 200. `wcop` is the both-feet whole-foot
centre of pressure in span half-lengths ahead of the mean MTP III joint
(x = 60 + 70 w mm), and a lean is the centre of mass's fore-aft shift from
the statue's.

1. **The committed stance reward pays for a small forward lean.** Its leg
   pose term is centred on the authored keyframe, which the servos cannot hold
   under the animal's weight, so moving toward the keyframe means leaning
   forward. Along the held leg-DC direction the committed reward peaks at a
   wcop offset of +0.332 from the statue (about 23 mm), every held lean with a
   pad statistic from 0.28 to 0.65 earns more than the statue, and a lean to
   0.746 (the gate's bar is 0.80) costs only 0.008 per step, −6.3 per window
   (`reward_landscape/out/derived.txt`; checked). Arm L's settled leg target,
   one of #624's options and off by default on `main`, moves the peak to
   +0.032, about 2 mm (the same file; checked).
2. **Under arm L the lean is not paid for, but it is barely resisted.** A
   held forward lean of 14 mm costs 0.013 per step (−10.7 per window), about
   0.3% of the statue's 4.08 per step, and the same lean backward 0.018 per
   step; nearly all of it is the settled leg pose term
   (`reward_landscape/out/derived.txt` and `held_tables.txt`; checked). The
   symmetry screen's arm L seed 60 drifted against its own reward: at 5M it
   scored 114.0 per episode below the arm L statue (10 deterministic seeds,
   9 full-horizon, one ending on head contact;
   `reward_landscape/out/ckpt_tables.txt`; checked).
3. **The lean lives in the ankles.** Over 51 panels of the symmetry screen's
   checkpoints, the ankles' part of the centre of mass's shift follows the
   shift with slope 1.00 (r 0.92), and each degree of ankle closure moves the
   centre of mass about 10.7 mm forward (the ankle angle changes −0.093° per
   mm, r −0.92), so the pad bar at 0.80 is about 4° of ankle from the statue
   (the pad statistic is 0.194 + 0.0142 per mm) (`onset/out/xstats.json`;
   checked). Over arm L seed 60's training evaluations from 1.05M steps the
   leg pose term fell as the signed centre of pressure rose, r −0.95
   (`onset/out/timeline.txt`; checked). At the term's width of 0.10 rad, 4°
   costs little.
4. **Other suspects look weaker** (as `recommendation.md` states them; not
   re-read here). The exploration noise falls smoothly and does not time the
   drift (`noise_robustness/work/training_noise_bins.txt`). The checkpoints
   picked by reward come from the first 1M steps and are fragile under their
   own noise (`onset/out/selection.txt`; §15.3 found the same). Of the physics
   changes rolled offline with policies trained on the unchanged plant, only a
   stiffer ankle spring set at the settled angle moved a drifted stance back;
   pulling the toe edge back made the statue fall, and the other foot changes
   did nothing for the lean (`mechanics/out/tables.txt`).
5. **A signed per-foot bar separates the drifted stances better than the pad
   rail.** Requiring each foot's whole-foot centre of pressure to lie
   [−5, +100] mm ahead of its MTP III, in place of the pad rail, fails 9 of
   seed 42's 29 clean episodes at 6M and 1 of seed 60's 40 (seed 3080),
   passes the one seed 52 6M episode the pad rail failed (seed 3063), and
   leaves the statue at 39 of 40 (`cop_statistic/out/analysis2.txt`;
   checked). The three runs below judge it as the screen-only criterion K2b;
   putting it in the gate would be a gate change, which is the maintainer's.

**Hypotheses, not findings.** (a) A heavier and narrower settled leg pose
term holds the stance where the drift starts: by arithmetic on the term's
form, which is an inference about gradients and not a training result, its
per-ankle restoring slope at weight 1.0 and tolerance 0.05 rad is 7.3 times
arm L's at 1°, 3.6 times at 3°, 1.9 times at 4°, about 0.9 times at 5° and
0.54 times at 5.8° (`recommendation.md`, Run 2, item 3), so it is a strong
centring price inside the gate's range and a weaker backstop once a seed has
leaned past about 5°.
(b) Nothing here explains what starts the drift, or why arm L seed 60 leaned
back by 6M; the runs test whether a price suppresses the drift, not why it
starts.

### 16.2 Decision D-D37: three full stance screen runs instead of walking pilot 2

The maintainer took D-D37 on 2026-10-18, superseding for these runs the rule
of 2026-10-16 (the short answer, §2 and §13: "no more stance-only reward
experiments unless walking points back to stance"); pilot 1 read STILL,
which by its criteria does not point back at the stance (§15.1). Walking
pilot 2 (§15.7) is parked: its branch `claude/pilot-walk-r9-b` and its
criteria are kept and nothing of it is trained.

| Run | Run id | Branch, commit | Configuration | Seed |
|---|---|---|---|---|
| 1 | `stance3_armL_s60` | `claude/screen-trex-r9-arml-12m`, `7e6b557d` | arm L as #624 ships it: the lateral lean term (weight 0.25, tolerance 0.015 m) and the settled leg target on the stance task, and `MirrorKLPPO` (kappa 0.3, sigma floor 0.01, mirrored critic); collapse reference 3981.4 | 60 |
| 2 | `stance3_armLT_s60` | `claude/screen-trex-r9-armlt`, `1af2d4d7` | arm L plus `leg_home_pose_weight` 1.0 (0.5) and `leg_home_pose_tolerance` 0.05 rad (0.10), "L+T"; collapse reference 4460.5 | 60 |
| 3 | `stance3_armLT_s52` | as Run 2 | as Run 2 | 52 |

- **What they share** (`criteria.txt`): physics r9, the stance stage through
  the SB3 notebook, 4 environments, 12,000,000 steps (the final checkpoint at
  12,001,280), the committed PPO schedules, gate and bars, an evaluation of 10
  episodes every 131,072 steps and a checkpoint every 524,288 steps, all
  kept. Both branches are built on #630's code (`04ced045`) with the cadence
  keys of `412f19ae` cherry-picked; neither is ever merged, and neither has a
  pull request (full commits in `colab_steps.txt`). The collapse references
  are production's 3756.8 plus each configuration's measured statue gain,
  224.6 under arm L and 703.7 under L+T: 3981.4 and 4460.5
  (`scratchpad/foot/stance3/statue/statue_L.json` and `statue_LT.json`;
  checked). The statues themselves score 3980.37 under arm L and 4459.40
  under L+T, each standing on 39 of 40 seeds, against 3755.72 re-measured
  for production (`refine_plan/out/statue_ref.json`; checked).
- **Runtime.** All three train on Colab L4 runtimes, the maintainer's choice of
  2026-10-18. `recommendation.md` recommended CPU, the only runtime the
  mirror-symmetric trainer had trained 12M on; `colab_steps.txt`, as sent,
  named CPU as the maintainer's choice of that day; both asked for one type
  for all three. The runs went on L4, and the maintainer kept L4 for all
  three. Run 1's `stage_config.json` on Drive, read on 2026-10-18, records
  commit `7e6b557d`, arm L's keys, the task `sha256:cd963518…`, the PPO
  hyperparameters `sha256:054ab483…` (`recommendation.md`'s expected value)
  and an NVIDIA L4, and it trains from 16:10 UTC; the run folder's
  `zero_action_baseline.json` has the statue standing at 3980.37 (39 of 40).
  Runs 2 and 3's `stage_config.json` files on Drive record commit
  `1af2d4d7`, seeds 60 and 52, collapse reference 4460.5, the two leg keys,
  the task `sha256:d4a3b8f8…`, the same PPO hyperparameters and an L4, from
  16:20 and 16:23 UTC; their `zero_action_baseline.json` files have the
  statue standing at 4459.40 (39 of 40).
- **The criteria, fixed before any training step** (`criteria.txt`, sha256
  `1ce910dcd6e2ff7b41a4f592b9f34b9a6c9cb398c523e88f0ebd19af754f18fa`): judged
  at 4,194,304, 5,242,880, 6,291,456, 8,388,608, 10,485,760 and the final, on
  the 40-episode panel (seeds 3042–3081) at heading 0. K1, the gate held:
  `stance_quality/v2` PASS at 6,291,456 and at the final, and at most one
  judged checkpoint below 37 of 40 clean and none below 33. K2, fore-aft held
  at every judged checkpoint: K2a, the pad rail over 0.80 on at most 3 of 40;
  K2b, screen only, each foot's signed whole-foot centre of pressure outside
  [−5, +100] mm of its MTP III on at most 3 of 40; K2c, screen only, no drift:
  the panel median within −60 to +30 mm of the statue's, and its range over
  the six checkpoints at most 30 mm. K3, robustness at 6,291,456 and the
  final: at most 2 of 20 joint-space noise falls and 1 of 20 own-noise falls.
  K4, roll at the same two: at most 3 of 40 full-horizon episodes over the
  tilt or corner-lift bar. A decision table fixed with them says what follows
  (L+T holding on both seeds is proposed for committing and certifying on two
  seeds; each other outcome names the next round), and the best final
  checkpoint of the three is offered as walking pilot 2's parent. Fails stay
  fails, and no run is stopped or changed on its readouts.
- **What they can show.** A screen certifies nothing: the configurations are
  not committed, so the official judge refuses the runs' task fingerprint,
  and the notebook's end-of-stage certificate on them is report only.
  Whether a screen run could ever count toward `certification_seeds` if its
  configuration were later committed byte-identically is not ruled; it is an
  open question for the maintainer (`recommendation.md` recommends no).
  `main` and its digests are unchanged.
- **Status:** launched on 2026-10-18; not yet judged. A later PR records the
  outcome, scored by the frozen `stance3/judge/score_stance3.py` from the
  judge files of `stance3/judge/judge_stance3.py`.

### 16.3 What changes in §13–§15

- **§13 and §15.7, walking pilot 2:** parked, not trained; its branch and
  criteria are kept, and the decision table above names its parent.
- **§13, "no more stance-only reward experiments":** superseded for these
  three runs by D-D37.
- **§14, this repository:** the folder `stance3/`, with its sources and sha256
  in `SOURCES.txt` and `SHA256SUMS`.
- **§14, branches:** `claude/screen-trex-r9-arml-12m` at `7e6b557d` and
  `claude/screen-trex-r9-armlt` at `1af2d4d7`; neither has a pull request.
- **§14, Google Drive:** the runs write `stance3_armL_s60/`,
  `stance3_armLT_s60/` and `stance3_armLT_s52/` under
  `MyDrive/mesozoic-labs/logs/tyrannosaurus_rex/anatomical/ppo/`.
- **§14, session scratch only:** the diagnosis' tables and scripts
  (`scratchpad/foot/tonight/`), and the branches' build record, smoke runs
  and judge validation (`scratchpad/foot/stance3/`).

## 17. Status 2026-10-19 (appended): the three stance screen runs judged

*Appended 2026-10-19; §1–§16 above are unchanged, so their "not yet judged"
is the wording of 2026-10-18.* The three runs of D-D37 (§16.2) trained to
their final 12,001,280 steps and were judged by their fixed criteria with the
frozen judge and scorer between 2026-10-18 20:45 and 2026-10-19 06:14 UTC.
This section records the verdicts, the decision-table line and the best final
checkpoint, which are results; the report-only readouts and the overnight
scratch experiment that followed; and what is left for the maintainer, which
is listed as proposals only. A screen certifies nothing: the verdicts are the
criteria's, not a gate's, nothing here certifies a run, writes a result bundle
or promotes anything, and `main` and its digests are unchanged. The evidence
is the folder `stance3/results/` added to
[stance_2026_10/](stance_2026_10/README.md) on 2026-10-19, which its own
[README](stance_2026_10/stance3/results/README.md) indexes; paths in
backticks below are relative to it, and every number below is read from
`run_log.txt`, the score files in `scores/` or the files named.

### 17.1 The judging

- **The runs.** All three trained to 12,001,280 steps on their Colab L4
  runtimes (`progress.json` "ended"; Run 2 last, at 04:58 UTC on
  2026-10-19), each keeping a checkpoint every 524,288 steps on Drive, as the
  criteria ask.
- **Fetch and binding** (`run_log.txt`). Each judged checkpoint was fetched
  read-only from Drive; each VecNormalize file was transcribed twice,
  independently, and the two copies were identical, and each zip passed
  `testzip`. The judge (`../judge/judge_stance3.py`, in each branch's clean
  clone) bound the plant (r9), the seed, the task, the symmetry map and the
  run's `stage_config.json` (commit, `n_envs`, hyperparameters, seed); every
  binding matched, with no hard failure. The checkpoints were judged as they
  arrived: 4,194,304 at 21:11–21:16 UTC, 5,242,880 at 22:30–22:35, 6,291,456
  at 23:39–23:45 (a container restart killed the first launch, which wrote
  nothing), 8,388,608 at 01:44–01:49, 10,485,760 at 04:00–04:05 and the
  final at 06:00–06:14.
- **The score.** `scores/score_final2.txt` and `.json`, written by the
  frozen scorer (`../judge/score_stance3.py`; the criteria's sha256
  `1ce910dc…` matches) at 08:17 UTC with the report-only extras judged; its
  verdict lines equal those of `scores/score_final.txt` (06:14 UTC). The
  committed scorer copy, run on 2026-10-19 on the judge files, reproduces it
  apart from the paths it names, and its tests pass 80 of 80.
  `checkpoints.csv` has one row per judge file.

### 17.2 The verdicts: all three runs fail

Per run, the judged checkpoints are 4,194,304, 5,242,880, 6,291,456,
8,388,608, 10,485,760 and the final (4.2M to 12M below); `x_c` is the panel
median of the both-feet whole-foot centre of pressure, in mm ahead of the MTP
III joints, against the statue's 64.0 mm on the same seeds.

- **Run 1, `stance3_armL_s60` (arm L, seed 60): fails K2 and K3.** K1
  passes: `stance_quality/v2` PASS at 6,291,456 (39/40) and at the final
  (38/40), and no judged checkpoint below 37/40. **K2 fails on K2c**: the
  offsets from the statue are −22.0, −32.6, −20.1, −28.0, −6.7 and −40.6 mm,
  all inside the [−60, +30] mm window, but `x_c` ranges from 23.4 to 57.3 mm,
  **33.9 mm** against the 30 mm bar (K2a and K2b pass, at most 2 per
  checkpoint). **K3 fails**: **7 and 11 of 20** joint-space noise falls at
  6,291,456 and the final (bar 2); own-sigma 1 and 0. K4 passes (0 and 0).
- **Run 2, `stance3_armLT_s60` (L+T, seed 60): fails K1 and K2, early.**
  Clean 0, 0, 36, 39, 39 and 40 of 40 from 4.2M to the final: **K1 fails**,
  with v2 FAIL at 6,291,456 and three judged checkpoints below 37/40 (two
  below 33). **K2 fails** from 4.2M to 6.3M: K2a 10 and 6 at 4,194,304 and
  5,242,880, K2b 14, 10 and 6 at 4,194,304, 5,242,880 and 6,291,456 (K2c
  passes: range 11.3 mm). K3 passes (0/0 at both) and K4 passes. The 0 of 40
  at 4.2M and 5.2M is actuator saturation (§17.4).
- **Run 3, `stance3_armLT_s52` (L+T, seed 52): fails K2.** K1 passes (v2
  PASS at 6,291,456, 40/40, and at the final, 39/40; one judged checkpoint
  below 37/40, 34/40 at 8,388,608, where one is allowed). **K2 fails three
  ways**: **K2a 6 at 8,388,608** (all six on the heel side), **K2b 4 at the
  final**, and the **K2c range 30.6 mm** (`x_c` 42.8 at 8,388,608 to 73.4 at
  the final, +9.3 mm from the statue). K3 passes (0/0 at both) and K4 passes.

**The recipe verdicts: all three runs fail.** None holds, and none holds
fore-aft. **The decision table's line is 3b**: "L+T fails K2 on a seed and
Run 1 also fails K2: a joint-space price does not hold the stance; the next
round tests a fore-aft term that needs code or the v6b ankle spring" (the
criteria name the ankle anchor at weight 0.4 and tolerance 0.10 rad, or the
COM band penalty with a neck guard). The line is the result; what the next
round is, if there is one, is the maintainer's decision (§17.6). Neither
configuration is proposed for committing by the table.

**The best final checkpoint by the fixed rule** (most clean panel episodes,
ties to fewer joint-space plus own-sigma noise falls) is **Run 2's final**
(`stage1_final`, sha256 `cc21604a…`): 40/40 clean, 0 joint-space and 0
own-sigma noise falls, K2a and K2b 0, `x_c` −10.2 mm from the statue. Run
3's final is next (39/40, 0 falls), then Run 1's (38/40, 11). On the
report-only extras it is clean everywhere: the fresh panel 7042–7081 40/40,
the forward-start tail panel 0 falls of 32, and the stage's six-heading
certificate, rolled offline, 40/40 at every heading. The criteria offer it as
the stance parent of the parked walking pilot 2; that is not decided (§17.6).

### 17.3 Report only: the extras

None of these enters a criterion (`checkpoints.csv`; `run_log.txt`).

- **3,145,728, judged like the J checkpoints:** Run 1 PASS 40/40 (`x_c`
  −22.0 mm), Run 2 FAIL 6/40 (saturation in 31; K2a 7, K2b 14), Run 3 PASS
  40/40 (−13.5).
- **Selection (S0–S3; clean of 40, own-noise falls, K2a and K2b).**
  `robust_best` and `best` are the same file in every run, saved at 2,752,512
  steps in Run 1 and 917,504 in Runs 2 and 3: 40/40, 0, 0 and 0 in all three.
  The final: Run 1 38/40, 0, 2 and 1; Run 2 40/40, 0, 0 and 0; Run 3 39/40, 0,
  1 and 4. The late-stable S3: Run 1 at 10,485,760 39/40, 0, 0 and 0; Run 2 at
  11,534,336 34/40, 0, 0 and 0; Run 3 at 4,194,304 38/40, 1, 2 and 2.
- **The fresh panel 7042–7081 at the final:** Run 1 PASS 39/40 (a fall on
  7047), Run 2 PASS 40/40, Run 3 PASS 39/40 (a fall on 7047).
- **The forward-start tail panel (32 seeds) at the final:** 6, 0 and 4 falls
  for Runs 1, 2 and 3; the statue 0 (the r9 arm L symmetry screen at 6M:
  seed 60 3, seed 52 7, seed 42 26).
- **The stage's six-heading certificate (`stance_quality_heading/v1`) on the
  finals, rolled offline in each branch's clone on the run's own task:** Run
  1 PASS 38/40 seeds clean at every heading (LCB 0.851), Run 2 PASS 40/40
  (0.928), Run 3 PASS 39/40 (0.887) (`certificates/`). **None is gate
  evidence**: the screen configurations are not committed, so the official
  judge refuses their task fingerprints.
- **The pre-registered readouts:** K5 flags no run; the late-fall fraction
  is above 5% only in the first 262,144-step bin (52.6%, 49.2% and 49.1%)
  and in Run 1's bin ending at 7,340,032 (5.1%).

### 17.4 What the readouts found (report only)

Three mechanisms, measured on rollouts that first reproduce the judge files
exactly and on the runs' evaluation ledgers; "inferred" marks what is not
measured.

1. **Arm L's whole-body ankle drift (Run 1).** At 4,194,304 Run 1's centre
   of mass had moved backward since about 2.3M steps, from −6 mm at 1M to
   −35 mm at 3–4.2M (the eval ledger's foot-site frame; the statue reads
   about −16 mm), and its signed whole-foot centre of pressure from
   +0.1..+0.3 to −0.40..−0.55. The ankle carries it: the centre of mass
   moves −10.6 mm per degree of ankle, −5.3 per degree of knee and −0.8 per
   degree of hip; from 2.3M it moved −14 mm per million steps while the ankle
   opened +1.43° per million, and the joint attribution predicts −32.2 mm
   against −34.5 mm measured. Its two falls at 4,194,304 (seeds 3059 and
   3063) are backward: the extrapolated centre of mass and then the centre
   of mass pass behind the heel line before any leg joint leaves its
   envelope by 5° (`readouts/r4194304_posture_notes.md`). Over the run it
   made three swings of 50–65 mm, carried by the ankle (up to +27 mm in one
   evaluation), the forward one between 6.16M and 6.55M; on the same
   seed under L+T (Run 2) the range after 3.15M was 14.3 mm against 64.7, the
   worst change over 1M steps −8.0 against +59.7 mm, and the ankle's range
   0.7° against 8.2° (`run_log.txt`; `readouts/drift_notes.md`). That drift
   is Run 1's K2c failure. It is the joint of the earlier forward lean (§7,
   §16.1), now in both directions.
2. **L+T's left-right foot split from the spawn stagger (Runs 2 and 3).**
   The reset noise (`reset_noise_scale` 0.05 on hip pitch and knee) starts
   the feet about 42–45 mm apart fore-aft (sd). The statue closes this to
   0.5 mm by step 999, mostly by yawing the pelvis square to the foot line;
   at 4,194,304 Run 3 kept 3.6 mm and Run 2 10.6 mm, with the rear foot
   toe-loaded and the forward foot heel-loaded: every K2b-outside episode
   has a stagger of at least 10.3 mm, and the correlation of the per-foot
   centre-of-pressure split with the stagger is −0.97. It is not marginal:
   with the pad bar at 0.90, K2a is still 6, and with K2b widened by 0.10
   still 13 (`run_log.txt`; `readouts/r4194304_saturation_notes.md` §3;
   `readouts/r4194304_k2_recount.txt`). Run 3's K2a failure at 8,388,608 is
   the same split, larger: the forward foot is heel-loaded in 39 of 40
   episodes, the policies keep 5–24 mm because their hip commands push into
   the twist (a right-left hip servo slope of +0.62 to +1.38 against the
   statue's −1.00), and at 8,388,608 only the policy turns toward the prey
   during the window (correlation 0.88). Its K2a needs the split and a
   common backward ankle opening together (+2.0°, the centre of pressure
   −0.14 against 6,291,456). On the fixed 8,388,608 policy, mirrored leg
   noise at reset cuts K2a from 6 to 3, the prey straight ahead to 4, and
   both to 1 (counterfactual rollouts, not the judge's task). The
   observation has no stagger or centre-of-pressure channel, but the
   stagger is a linear function of the observed leg angles
   (`readouts/split_notes.md`). Under arm L, Run 1 had the largest stagger
   at 4,194,304 (sd 21.2 mm) but almost no split: its ankle difference
   tracks the stagger and keeps the soles parallel
   (`readouts/r4194304_saturation_notes.md` §3).
3. **Run 2's tail-actuator saturation.** Run 2's 0 of 40 at 4,194,304 and
   5,242,880 is the gate's `max_actuator_saturation_fraction` (40 and 39
   episodes). One actuator, `tail_3_pitch_act`, is pinned at +1 in 40 of 40
   episodes (the fraction at 0.99 is 0.79–1.00, mean 0.96; the raw mean
   +1.05); no other actuator in any run exceeds 0.10. It is a position servo
   (kp 60, joint and control range ±0.209 rad, force range ±50): its target
   sits at the stop while the joint sits at −0.007 rad with 12.9 N·m, 0.26 of
   the limit, so the command is pinned and the force is not; the statue's
   tail_3 hangs at −0.093 rad with 5.6 N·m. The L+T leg term does not cover
   the tail, and Run 2's legs sit within 0.005 rad of their settled targets.
   It came and went. In the evaluation ledger it held 6–10 of 10 episodes
   from 2.23M to 6.03M and was gone at 6.16M (2 of 40 panel episodes at
   6,291,456); it returned in 10 and 8 of 10 at 7.21M and 7.34M, in 8–10 of
   10 from 9.31M to 10.09M (28 of 40 panel episodes over the bar at
   9,961,472) and in 9, 10 and 9 of 10 from 10.88M to 11.14M (5 at 11.27M;
   6 of 40 panel episodes at 11,534,336, the S3 pick); 39 of the run's 91
   evaluations have 8 or more of 10 saturated, and the final's panel is
   clean (the run's `eval_episodes.csv`, counted for this record, session
   scratch only; `checkpoints.csv`). The gate's threshold is 0.99 on clipped
   actions (`SATURATION_ABS`); the env's 0.9 is the reward ramp
   (`readouts/r4194304_saturation_notes.md`; `run_log.txt`). Inferred: a
   tail habit of this seed, not line 4's "dose too strong" on the legs (K2
   failed in any case).

Also measured: at 4,194,304 every run's hip-pitch, knee and ankle
`exp(log_std)` (0.037–0.044, 0.017–0.019 and 0.022–0.024) is below the 0.08
floor that the criteria's line 2 would add.

### 17.5 The overnight scratch experiment (scratch only, report only)

Between 02:10 and 04:30 UTC on 2026-10-19 the split and drift diagnoses above
designed three continuations, with their readouts and the lines that would
support or refute each hypothesis fixed before launch
(`night_scratch/spec_section7.md`). Each ran locally on CPU from a Colab
8,388,608 checkpoint with one candidate reward term on, from a scratch clone
(`80328abf`, branch `scratch/stance3-night`; never pushed, never merged; the
terms off are bitwise identical to the screen): **A**, Run 3 plus a
foot-stagger penalty (weight 0.3, tolerance 45 mm); **B**, Run 3 plus the
ankle anchor (weight 0.4, tolerance 0.10 rad, line 3b's named candidate at its
pre-registered dose); **C**, Run 1 plus the same stagger penalty. Two
container restarts interrupted them, and the second left the pre-registered
fallback endpoint, E = 9,961,472. The matched no-change controls are the
Colab runs' own checkpoints at E, fully bound (their VecNormalize files,
transcribed twice and compared as for the other fetched checkpoints, are not
checksum-verified against Drive). The arms' judge files are relaxed-binding,
so the scorer refuses them and none enters K1–K4. The readouts are
`night_scratch/readouts_night.md` and `readouts_control.md`; the
interpretation as written is `night_scratch/interpretation.md`, and
`night_scratch/verify_result.json` is the adversarial verifier's check of it,
whose corrections are applied here.

- **The anchor (H_B) is refuted** by its own pre-registered line: the ankles
  went from +2.00° to +0.13° past settled and the common centre of pressure
  from −0.253 to −0.039, yet K2a went from 6 to 10 and K2b from 3 to 13 (the
  split's sd from 0.685 to 1.029). At this dose, on this checkpoint, pricing
  the common ankle opening did not bring K2a to 3 or below. Not claimed: that
  K2a rose in a meaningful sense (6 against 10 of 40 is within panel noise,
  Fisher p 0.40); that it made the stance worse than no change (that rests
  on the panel alone, 10 against the control's 0; on the eval ledger its
  pad-over count is no worse than the control's); or that the refutation
  holds whatever n is, or refutes the two-part recombination model (H_B's
  premise that the split stays did not hold).
- **The stagger term (H_A, on Run 3) is inconclusive against the matched
  control.** Its outcome rows sit on the supports side: the stagger's sd went
  from 20.9 to 7.5 mm, the split's from 0.685 to 0.471, K2a, K2b and roll from
  6, 3 and 3 to 0, 0 and 0, 40/40 clean, no noise falls and no safety flag.
  But the mechanism it names is not shown: the right-left hip slope is +0.52
  (95% seed-bootstrap interval 0.36 to 0.70) against the refute bar of +0.5;
  and the no-change control also reached a clean panel at E (stagger sd 9.7
  mm, K2a and K2b 0), so the E panel cannot separate them (the stagger's sd
  differs by −2.2 mm, interval −4.8 to +0.7). Attribution to the term is not
  established, neither shown nor excluded. The pre-registered eval ledger
  favours A over the control (pad-over 6 of 120 against 29 of 120), confounded
  by harness and evaluation seeds. The control's recovery did not last: Run
  3's final relapsed to K2b 4.
- **The stagger term on Run 1 (H_C) is refuted**: the stagger's sd went from
  29.3 to 22.6 mm against the refute bar of 20 or more, beside the control's
  24.2 (difference −1.6 mm, interval −5.2 to +2.4). C's two panel falls, on
  seeds 3046 and 3063, are also in the control at E.
- **The fixed reading rules** (spec §7.3): no reading applies cleanly; the
  nearest are reading 1's "C refutes" branch and reading 4 in substance for
  A, and both lead to the same proposals (§17.6).
- **Also seen at E:** Run 2's control relapsed into saturation (12/40 clean,
  28 episodes over the bar), as in §17.4.

The spec's own follow-up, a local no-change continuation of Run 3 from
8,388,608 to 9,961,472 (`night_R3_nochange`), was launched at 08:15 UTC on
2026-10-19; its outcome is in §17.8.

### 17.6 Pending: proposals for the maintainer

Nothing below is decided; each item is the maintainer's.

1. **The next screen's design.** Line 3b names candidates fixed before
   training: a fore-aft term that needs code (the ankle anchor at weight 0.4
   and tolerance 0.10 rad, or the COM band penalty with a neck guard) or the
   v6b ankle spring. The overnight experiment bears on them at n = 1, on
   continuations rather than runs from scratch: the anchor at that dose is
   refuted on its own line, and the stagger term is unattributed. Its
   interpretation and verifier propose: commit neither term; run the local
   no-change continuation before any Colab arm; test the stagger term, if at
   all, from scratch beside an unchanged L+T arm on the same seeds, with at
   least 2 seeds per arm; drop the anchor at this dose; use L+T rather than
   arm L as the base; and put an unchanged arm in any screen and judge more
   than one checkpoint, since a single checkpoint swung between PASS and
   FAIL within 0.5M steps. Inferred, not tested: the v6b spring acts on the
   same common ankle opening as the anchor.
2. **Walking pilot 2's parent.** The fixed rule picks Run 2's final (§17.2),
   and the criteria offer it as the parent. Pilot 2 is parked, and its own
   criteria (§15.7) name r9 arm L seed 60 at 6M (`70f3a444…`) as its
   parent, so another parent would need new pilot 2 criteria. The overnight
   drift diagnosis, before the finals were judged, named Run 3 at 6,291,456
   as the best judged checkpoint so far, by another rule. Whether pilot 2
   runs, and from which parent, is open.
3. **Criteria and telemetry.** Proposed in the readouts and the
   interpretation: a saturation alarm on the eval-ledger read (39 of Run 2's
   91 evaluations had 8 or more of 10 episodes saturated, in four spells
   from 2.23M to 11.14M (§17.4), which `eval_episodes.csv` already records
   and nothing watched); the side-door checks (touchdown,
   all-feet support, settle touchdowns, yaw change, roll, saturation, reward
   ratio, the `x_c` window, the hop watch and the collapse stop) kept as
   pre-registered flags; the stagger and the ankle opening as readouts (the
   ankle opening led Run 3's slip by 12–13 evaluations, the centre of mass
   by 5–6, while the clean and pad counts lagged); and mirrored spawn noise
   and the prey straight ahead as diagnostic panels only, not in the
   training task. Still open from §16: the signed per-foot whole-foot bar
   (K2b) in the gate, a gate change (§16.1 item 5), and whether a screen run
   could ever count toward `certification_seeds` (§16.2).

### 17.7 What changes in §13–§16

- **§13 and §16.3, the three runs:** judged; all three fail, and the
  decision table's line is 3b. No T. rex stance is certified, and the
  two-seed ask stands. Walking pilot 2 stays parked, its branch and criteria
  kept.
- **§14, this repository:** the folders `stance3/results/` and
  `stance3/night/` (§17.8), with their sources and sha256 in `SOURCES.txt`
  and `SHA256SUMS` (now 321 files).
- **§14, branches:** unchanged; `claude/screen-trex-r9-arml-12m` at
  `7e6b557d` and `claude/screen-trex-r9-armlt` at `1af2d4d7`, neither with a
  pull request. The overnight experiment's branch `scratch/stance3-night`
  (`80328abf`) exists only in the session's scratch.
- **§14, Google Drive:** the three runs' folders `stance3_armL_s60/`,
  `stance3_armLT_s60/` and `stance3_armLT_s52/` under
  `MyDrive/mesozoic-labs/logs/tyrannosaurus_rex/anatomical/ppo/` hold their
  checkpoints; each judged one's sha256 is in `checkpoints.csv`.
- **§14, session scratch only:** the judge files
  (`scratchpad/foot/stance3/judged/`), the runs' fetched evaluation and
  training ledgers (`scratchpad/foot/stance3/runs/`), the readouts'
  rollouts, tables and scripts, the certificates' per-episode files, and the
  overnight runs, their judge files and the verifier's scripts
  (`scratchpad/foot/stance3/night/`).

### 17.8 Night continuations (report only)

*Added 2026-10-19, later.* The spec's local no-change control of §17.5
(`night_R3_nochange`: Run 3 from the Colab 8,388,608 checkpoint with no
added term, in the same scratch clone and harness as A and B) reached
9,961,472 at 09:59 UTC, after a container restart and a resume from
9,437,184 (A and B resumed from 8,912,896; inferred to add noise, not bias).
It was judged, relaxed-binding as A and B, and rolled at 9,437,184 and
9,961,472; each rollout set reproduces its judge file on all 40 seeds. None
of it enters K1–K4. The comparison is `../night/nochange_compare.md`;
`../night/interpretation.md`, which it cites, is the same file as
`night_scratch/interpretation.md`.

At 9,961,472, on the 40-seed panel (measured):

| Continuation | Clean of 40 | K2a | K2b | Window stagger sd (mm) |
|---|---|---|---|---|
| A, Run 3 plus the stagger term | 40 | 0 | 0 | 7.5 |
| B, Run 3 plus the ankle anchor | 30 | 10 | 13 | 18.1 |
| Local no-change | 25 | 15 | 6 | 25.9 |
| Colab no-change (§17.5's control) | 40 | 0 | 0 | 9.7 |

- **H_B (the ankle anchor) stays refuted** by its own pre-registered line
  (§17.5). Against the local no-change run, B is worse only on K2b (13
  against 6; exact McNemar p 0.016) and better on K2a (10 against 15) and
  on the stagger.
- **H_C (the stagger term on Run 1) stays refuted** (§17.5). The
  comparison adds no Run 1 continuation.
- **H_A (the stagger term on Run 3) stays inconclusive.** The local
  no-change run did not recover, so "the unchanged recipe did what A did"
  does not hold in this harness: A minus the local run is −18.5 mm of
  stagger sd (95% paired seed-bootstrap interval −23.3 to −12.9) and −15
  K2a episodes (−21 to −9). But the Colab no-change run, from the same
  checkpoint to the same step, differs from the local one by as much:
  −16.3 mm (−22.8 to −9.1) and −15 K2a (−21 to −9). A is level with the
  Colab run: −2.2 mm (−4.9 to +0.5), and K2a and K2b 0 against 0.

**Why attribution is unclear.** There is one run per arm (n = 1). The
bootstrap intervals cover only which 40 spawns the panel draws, not the
run-to-run variance of training, and the two no-change runs from one
checkpoint show that variance to be as large as A's apparent effect.
Inferred, not tested: A is the only one of the four continuations whose
stagger stayed closed at both judged checkpoints and in every ledger
evaluation after 8.91M (5.6–10.9 mm per evaluation, against 11.4–43.5 mm
for the local no-change run and 10.2–51.9 mm for the Colab one), which
favours the term at n = 1. More seeds of both arms, not a longer single
run, would settle it.

## 18. Status 2026-10-19, later (appended): decision D-D39, the stance4 screen

*Appended 2026-10-19; §1–§17 above are unchanged.* After §17's verdicts the
maintainer took decision D-D39 on 2026-10-19
([BEHAVIOR_RECIPES_PLAN.md](../BEHAVIOR_RECIPES_PLAN.md) §6.2): another
T. rex stance screen, stance4, that tests the foot-stagger penalty of §17.5
and §17.8 from scratch. It takes item 1 of §17.6 and parts of items 2 and 3.
Its criteria were fixed before any training step:
[stance_2026_10/stance4/criteria.txt](stance_2026_10/stance4/criteria.txt),
sha256 `978a93da8da469c1acde7b9f7261d9236e4799eb2111b3af0f0cbeaa153f0f38`.
They replace a first file that named a Colab CPU runtime, as the decision
did; they name an L4 GPU runtime, for the maintainer to confirm (§18.1). As
a screen it certifies nothing.

### 18.1 The runs and the control

- **Arm LTS ("L+T + stagger"):** §16.2's L+T recipe (branch
  `claude/screen-trex-r9-armlt` at `1af2d4d7`) plus the stance stagger
  penalty. After the settle (step > 200), while the command is zero, each
  step pays −0.30 × (1 − exp(−(s / 0.045)²)), where s is the right foot
  site's lead over the left along the gravity-aligned pelvis heading
  (`stance_stagger_weight` 0.30, `stance_stagger_tolerance_m` 0.045). s
  enters squared, so the term is mirror-invariant. The statue's mean |s| in
  the window is about 2 mm (measured; the criteria).
- **The branch** `claude/screen-trex-r9-armlt-stagger`: `1af2d4d7` plus
  `ff7114c0` (the term as an opt-in `[env]` option, off by default; off, it
  leaves the L+T task digest `d4a3b8f8…` unchanged) and `c1e146a8` (the two
  keys on in the stance config; the task `sha256:3fa46708…`; the collapse
  reference 4459.3 against L+T's 4460.5). It is never merged and has no pull
  request.
- **The runs:** `stance4_LTS_s52`, `stance4_LTS_s60` and `stance4_LTS_s42`
  (seeds 52, 60 and 42), from scratch on physics r9, 12,000,000 steps (final
  12,001,280) with the SB3 notebook, `n_envs` 4 and the committed PPO
  schedules; an evaluation of 10 episodes every 131,072 steps and a
  checkpoint every 524,288 steps, all kept, as for §16's runs.
- **The runtime: CPU in the decision, L4 in the criteria.** The maintainer's
  decision named Colab CPU runtimes, as stance3's Colab steps had. Measured,
  stance3's runs trained on Colab L4 GPU runtimes: the `gpu` block of each
  run's `01_stance/stage_config.json`, in the copies fetched for the judging
  at 4,194,304 steps, reads `L4` (§16.2). Each copy records only the session
  that wrote it, and its run block has no load keys (measured), so that
  session started the run. A resume would rewrite the file, and none is
  recorded, so that each control run stayed on L4 throughout is inferred.
  Before any of the three runs started, the session preparing them re-fixed
  the criteria for L4, the control's runtime type, and told the maintainer.
  The Colab steps ask the maintainer to choose L4 and to report each run's
  runtime type at its start. The maintainer's confirmation is pending; a
  later PR records it with the runs. That the networks ran on the GPU is
  inferred, not measured: the trainer gives SB3 no device, and SB3's default
  picks CUDA when it finds one. If no L4 is available, the maintainer may
  pick another runtime. Each run's runtime is read from the `gpu` block of
  its `stage_config.json` (no block means a CPU runtime) and reported beside
  the verdicts; the runtime never changes a verdict.
- **The first criteria file named a CPU runtime.** That file (sha256
  `8bda933e682955e36e0f4bafcd184df9ed1df1238ae1a92479dfebead04caf9e`) said
  "a Colab CPU runtime", as the decision did. The two files differ only in
  the runtime (measured: line 7 changed and two lines added). The first file
  stays in the session's scratch (`scratchpad/foot/stance4/superseded/`).
  Measured, the r10 screen's `01_stance/stage_config.json` files (§6; seeds
  42, 52 and 60) also record an L4, so §16.2's statement that CPU was the
  only runtime the mirror-symmetric trainer had trained 12M on is wrong, as
  is the same claim in `stance3/colab_steps.txt`; both are left unedited.
- **The control is not retrained.** It is §16's Run 2 (`stance3_armLT_s60`)
  and Run 3 (`stance3_armLT_s52`), as judged in §17.2: Run 2 fails K1 and
  K2 and passes K3 and K4; Run 3 fails K2 and passes K1, K3 and K4. The term
  is the only task difference between the arms. The control trained a day
  earlier, not beside the stagger runs, and seed 42 has none; the criteria
  name both as caveats. Not retraining it departs from the recommendation in
  §17.6 item 1 of an unchanged L+T arm in the same screen on the same seeds.

### 18.2 Criteria, credit rule and decision table

- **K1–K4 exactly as §16.2's**, with the same judge settings, the same six
  judged checkpoints (4,194,304, 5,242,880, 6,291,456, 8,388,608,
  10,485,760 and the final), the same HOLDS (K1–K4 all hold) and FORE-AFT
  HOLDS (K1 and K2), and K2c's 30 mm range bar unchanged.
- **The credit rule.** The term is adopted, that is proposed to the
  maintainer for committing, only if (a) at least 2 of the 3 stagger runs
  hold, and (b) for each of K1–K4 the fraction of stagger runs passing it is
  at least the control's. With the control's results (K1 1/2, K2 0/2, K3
  2/2, K4 2/2) and (a), (b) means that no stagger run fails K3 or K4.
- **The decision table** (H, the stagger runs that hold; K2p, those passing
  K2):
  1. H ≥ 2 and (b): adopt. Propose the option (off by default) and the two
     keys as the T. rex stance recipe, with the best holding final as the
     stance parent.
  2. H ≥ 2 without (b): not adopted; replicate on 2 more seeds before any
     commit.
  3. H = 1: not adopted; not separated from seed luck; replicate on 2 more
     seeds before any commit.
  4. H = 0 and K2p ≥ 2: not adopted; the next round targets the other
     failures (for K3, the leg exploration-noise floor that line 2 of
     stance3's decision table names, §17.4).
  5. H = 0 and K2p ≤ 1: not adopted; the next round tests a per-foot
     centre-of-pressure band term (new contact-reward code) or the v6b ankle
     spring.
- **Walking pilot 2's parent, in every case:** the best stagger final by
  §17.2's rule is compared with Run 2's final (40/40 clean, no noise falls;
  in the maintainer's words, "the current walking pilot 2 parent"), and the
  better is offered. Pilot 2 stays parked until the maintainer says
  otherwise.
- Fails stay fails, and no run is stopped or changed on its readouts.

### 18.3 What else D-D39 settles

- **Dropped from further screens:** the ankle anchor (§17.5, §17.8) and arm
  L (§17.2, §17.4).
- **Report only, never in a verdict:** every 1,048,576-step checkpoint
  outside the six, judged like them, for the stagger runs and for the
  controls where fetched; diagnostic panels at each judged checkpoint for
  both arms (mirrored leg noise, the prey straight ahead, both); §16's
  pre-registered readouts; a saturation alarm on the eval ledger (two
  evaluations in a row with at least 5 of 10 episodes listing
  `max_actuator_saturation_fraction`, §17.6 item 3); and stagger watches
  (the episode yaw change, training-fall causes with tail and head contact,
  and the stagger at step 200 against the window).
- **Still open:** the signed per-foot bar in the gate (§16.1 item 5) and
  whether a screen run could ever count toward `certification_seeds`
  (§16.2).
- The runs had not started when this was recorded. A later PR records their
  outcome.

### 18.4 What changes in §13–§17

- **§13, §16.3 and §17.7:** the next round is decided (D-D39). Walking
  pilot 2 stays parked, its branch and criteria kept.
- **§14, this repository:** `stance4/criteria.txt`, byte-identical to the
  file fixed before training, with its source in `SOURCES.txt` and its
  sha256 in `SHA256SUMS` (now 322 files).
- **§14, branches:** `claude/screen-trex-r9-armlt-stagger` at `c1e146a8`,
  never merged, with no pull request.

## 19. Status 2026-10-20 (appended): the stance4 screen judged

*Appended 2026-10-20; §1–§18 above are unchanged, so §18's "not yet trained"
and "pending" are the wording of 2026-10-19.* The three runs of D-D39 (§18)
trained to their final 12,001,280 steps on Colab L4 GPU runtimes and were
judged by their fixed criteria with the stance4 judge and scorer on
2026-10-20, the 1,048,576-step checkpoints from 02:47 UTC and the verdict of
record at 14:08 UTC. This section records the verdicts, the decision-table
line and the parent comparison, which are results; the report-only extras, the
notebook's own gates, the diagnostic panels and the readouts; and two
overnight scratch experiments, report only (§19.6 and §19.7). The maintainer's
decisions that followed, on 2026-10-20, are logged in short in the last lines
of `results/run_log.txt` (those beginning "MAINTAINER DECISION") and recorded
in §20. A screen certifies nothing: the verdicts are the criteria's, not a
gate's, nothing here certifies a run, writes a result bundle or promotes
anything, and `main` and its digests are unchanged. The evidence is in the
folder `stance4/` of [stance_2026_10/](stance_2026_10/README.md), with the
subfolders `results/` (its own
[README](stance_2026_10/stance4/results/README.md) indexes it), `night/`,
`night2/` and `copband/`, added on 2026-10-20; paths in backticks below are
relative to `stance4/`, and every number below is read from
`results/run_log.txt`, the score files in `results/scores/` or the files
named. A verification pass over the session's human-facing report on these
runs (a document outside the repository, not copied) found 25 statements
over-broad or unsupported, all corrected there (`results/run_log.txt`); this
section uses the corrected forms.

### 19.1 The runs and the judging

- **The runs.** `stance4_LTS_s52` started training at 23:58 UTC on 2026-10-19,
  `stance4_LTS_s60` and `stance4_LTS_s42` at 00:01 and 00:04 UTC on 2026-10-20
  (when each wrote its `stage_config.json`). Each was read from Drive at its
  start (read-only): commit `c1e146a8`, not dirty, the run's seed, `n_envs` 4,
  the two stagger keys (0.30 and 0.045 m), the L+T and symmetry keys, the
  cadence, the collapse reference 4459.3, the PPO hyperparameters `054ab483…`,
  the task `sha256:3fa46708…`, and a zero-action baseline standing at 4458.18
  (39 of 40 standing). All three trained to 12,001,280 steps: s42's and s60's
  final checkpoints were saved at 12:08 and 12:10 UTC, and s52's
  `progress.json` on Drive read "ended" at 12:29 UTC. Each kept a checkpoint
  every 524,288 steps on Drive, as the criteria ask.
- **The runtime, measured.** The `gpu` block of each run's
  `01_stance/stage_config.json` reads an NVIDIA L4 (23.7 GB, CUDA 13.0), the
  runtime the criteria name. That file records only the session that wrote it.
  The `progress.json` files of s60 and s42 record one session each, from 00:02
  to 12:10 UTC and from 00:04 to 12:08 UTC, so that those two stayed on the L4
  throughout is inferred; s52's `progress.json` was not fetched (its status,
  ended at 12,001,280 steps, was read from Drive's metadata). The maintainer
  confirmed the L4 for all three on 2026-10-20 (the run log's last line;
  §20.1).
- **Fetch and binding.** Each judged checkpoint was fetched read-only from
  Drive; each VecNormalize file was transcribed twice, independently, and the
  copies compared, with a third transcription where the two differed (the log
  records two: s60 at 4,194,304 and s52 at 9,437,184). The judge
  (`judge_stance4.py`, sha256 `3c8c0759…`, in the session's scratch only)
  bound the plant (r9), the seed, the task, the symmetry map and the run's
  `stage_config.json` (commit, seed, `n_envs`, hyperparameters, environment
  and symmetry keys, gate bars); in all 51 judge files every binding matched,
  with no hard failure and no relaxation (the judge files, session scratch
  only). The six judged checkpoints (J) were judged as they arrived: 4,194,304
  by 05:46 UTC, 5,242,880 by 06:56, 6,291,456 by 09:30 (the container was
  reclaimed at about 07:35, before the first launch ran), 8,388,608 by 10:56,
  10,485,760 by 11:52, and the finals by 13:26 (s60, s42) and at 14:07 (s52).
- **The score.** `results/scores/score_final.txt` and `.json`, written at
  14:08 UTC by `score_stance4.py --runs-ended` (sha256 `0d35e540…`, session
  scratch only), the verdict of record: it loads stance3's frozen scorer
  unchanged (`../stance3/judge/score_stance3.py`, sha256 `2c3879d3…`), extends
  only its judge-identity constants, and checks the criteria's sha256
  (`978a93da…`, matched). Re-run on the judge files, it reproduces the record
  apart from the path it names and s52's `robust_best` and `best` cells, which
  were judged after it (`best` at 14:13 and `robust_best` at 14:18 UTC).
  `results/checkpoints.csv` has one row per judge file (51), recomputed with
  the frozen scorer's own counts (40 cells agree with the score).

### 19.2 The verdicts: all three runs fail

Per run, the judged checkpoints are 4,194,304, 5,242,880, 6,291,456,
8,388,608, 10,485,760 and the final (4.2M to 12M below); `x_c` is the panel
median of the both-feet whole-foot centre of pressure, in mm ahead of the MTP
III joints, against the statue's 64.0 mm on the same seeds.

- **Run 1, `stance4_LTS_s52` (seed 52): fails K1 and K2.** Clean 28, 28, 21,
  23, 23 and 39 of 40 from 4.2M to the final. **K1 fails**:
  `stance_quality/v2` FAIL at 6,291,456 (PASS at the final), and five judged
  checkpoints below 33 of 40. **K2 fails**: K2a 12, 12, 19, 17 and 17 from
  4.2M to 10.5M, and K2b 16, 20, 20, 20 and 21 there and 4 at the final (K2c
  passes: `x_c` 58.6 to 65.2 mm, a range of 6.6 mm). K3 passes (0 and 0 of 20
  joint-space and own-sigma noise falls at 6,291,456 and the final) and K4
  passes (roll rail 1 and 0 of 40).
- **Run 2, `stance4_LTS_s60` (seed 60): fails K1 and K2.** Clean 29, 29, 30,
  27, 30 and 39 of 40. **K1 fails**: v2 FAIL at 6,291,456 (PASS at the final),
  and five judged checkpoints below 33 of 40. **K2 fails**: K2a 11, 11, 10, 13
  and 10 from 4.2M to 10.5M, and K2b 16, 13, 16, 13 and 5 (the final 0 and 0;
  K2c passes: `x_c` 43.2 to 58.2 mm, a range of 15.0 mm). K3 passes
  (joint-space 0 and 1 of 20, own-sigma 0 and 0) and K4 passes (0 and 1 of
  40).
- **Run 3, `stance4_LTS_s42` (seed 42): fails K1–K4.** Clean 0, 3, 0, 3, 17
  and 14 of 40: in most panel episodes up to the final the gate's
  actuator-saturation rail failed (§19.5 item 3). **K1 fails**: v2 FAIL at
  6,291,456 and at the final, and all six judged checkpoints below 33 of 40.
  **K2 fails**: K2a 4 at 6.3M and 5 at the final, and K2b 6, 10, 10 and 5 at
  5.2M, 6.3M, 10.5M and the final (K2c passes: a range of 7.4 mm). **K3
  fails**: 8 of 20 joint-space noise falls at the final (bar 2; own-sigma 0).
  **K4 fails**: 4 of 40 roll-rail failures at the final (bar 3).

Per K test, against the control (§18.1: stance3 Runs 2 and 3, as judged in
§17.2, not retrained; the scorer reproduces their verdicts from their judge
files):

| Test | s52 | s60 | s42 | Stagger runs passing | Control passing |
|---|---|---|---|---|---|
| K1, the gate held | FAIL | FAIL | FAIL | 0 of 3 | 1 of 2 (Run 3) |
| K2, fore-aft held | FAIL | FAIL | FAIL | 0 of 3 | 0 of 2 |
| K3, robustness | PASS | PASS | FAIL | 2 of 3 | 2 of 2 |
| K4, roll | PASS | PASS | FAIL | 2 of 3 | 2 of 2 |

**The recipe verdicts: all three runs fail; H = 0 and K2p = 0.** None holds,
and none holds fore-aft. The credit rule fails on both parts: (a) no stagger
run holds, and (b) the arm passes K1, K3 and K4 less often than the control.
**The decision table's line is 5, final**: "H == 0 and K2p <= 1: not adopted;
the stagger penalty does not hold fore-aft from scratch; the next round tests
the per-foot CoP band term (new contact-reward code) or the v6b ankle spring
(stance3 line 3b)". The line was fixed at 4,194,304, where all three runs had
a judged checkpoint below 33 of 40 (K1) and s52 and s60 failed K2, so no later
checkpoint could change it; the formal reading waited for the runs to end. The
stagger penalty is not adopted. Which round comes next is the maintainer's
decision: D-D40, a from-scratch screen of a per-foot centre-of-pressure band
(§20.2).

**The parent comparison** (stance3's rule: most clean panel episodes at the
final, ties to fewer joint-space plus own-sigma noise falls): **the best
stagger final is Run 1's, s52's** (`stage1_final`, sha256 `c83aeb04…`: 39 of
40 clean, 0 noise falls, K2a 1, K2b 4, `x_c` −2.2 mm from the statue), then
s60's (39 of 40, 1 fall) and s42's (14 of 40, 8). It is one episode short of
stance3 Run 2's final (`cc21604a…`: 40 of 40, 0 falls), so **the incumbent
stays**: the criteria offer stance3 Run 2's final as walking pilot 2's parent,
as before. No stagger final holds, so line 1's stance parent does not arise.
Pilot 2's parent is decided in §20.4.

### 19.3 Report only: the extras

None of these enters a criterion (`results/checkpoints.csv`;
`results/scores/`; `results/extras.txt`).

- **The 1,048,576-step checkpoints outside the six, judged like them:** at
  1,048,576 all three PASS (40, 39 and 39 of 40 for s52, s60 and s42, no noise
  falls, `x_c` within 4 mm of the statue). At 2,097,152, 3,145,728, 7,340,032,
  9,437,184 and 11,534,336: s52 33, 35, 27, 24 and 28 of 40 (K2a 7, 5, 13, 16
  and 12); s60 37, 35, 31, 39 and 40 (K2a 3, 5, 9, 1 and 0; 3 panel falls at
  2,097,152; PASS at 9,437,184 and 11,534,336); s42 5, 1, 0, 5 and 16.
- **Selection (S0–S3; clean of 40, own-noise falls, K2a and K2b).** s52:
  `robust_best` (917,504 steps) and `best` (524,288) 40/40, 0, 0 and 0 both,
  judged after the score was written. s60: `robust_best` and `best` are one
  file, saved at 2,752,512: 33/40, 0, 5 and 9. s42: `robust_best` (917,504)
  38/40 and `best` (524,288) 39/40, both 0, 0 and 0. The finals: s52 39/40, 0,
  1 and 4; s60 39/40, 0, 0 and 0; s42 14/40, 0, 5 and 5. The late-stable S3 is
  s42's 8,388,608 (3/40); s52 and s60 have none (no periodic checkpoint met
  its rule).
- **The fresh panel 7042–7081 at the final:** s52 PASS 39/40 (0 falls), s60
  PASS 38/40 (1 fall), s42 FAIL 14/40 (1 fall); stance3 Run 2's final 40/40
  and Run 3's 39/40.
- **The forward-start tail panel (32 seeds) at the final:** 2, 6 and 5 falls
  for s52, s60 and s42; the statue 0, stance3 Run 2's final 0 and Run 3's 4.
- **The common-noise panel (40 seeds) at the final:** 5, 1 and 30 falls; no
  control reference was measured.
- **The pre-registered readouts:** K5 flags no run. The late-fall fraction is
  above 5% only in the first 262,144-step bin (50.2%, 48.7% and 49.1%). The
  saturation alarm (two evaluations in a row with at least 5 of 10 episodes on
  the saturation rail) fires on s42 (§19.5 item 3) and on neither s52 nor s60
  (0 of 91 evaluations each).

### 19.4 Report only: the notebook's gates and the diagnostic panels

- **The notebook's own end-of-stage gate**, `stance_quality_heading/v1`,
  judged each run's `robust_best` checkpoint at six headings
  (`results/notebook_gates/`, trimmed): s52 PASS, 40/40 seeds clean (LCB
  0.928, at 917,504 steps); s60 FAIL, 33/40 (0.696, at 2,752,512); s42 PASS,
  38/40 (0.851, at 917,504). Each `checkpoint_sha256` equals the `robust_best`
  row's in `results/checkpoints.csv` (measured). These are not the screen's
  verdicts: the gate judged an early checkpoint, not the finals the criteria
  judge, and as screen runs none is gate evidence.
- **The diagnostic panels** (`results/panels_summary.csv`): the judge's panel
  with the legs mirrored at reset, with the prey straight ahead, and with
  both, at every J checkpoint of the three runs and of the two controls (and
  the unchanged panel at every J checkpoint of the stance4 runs and at
  8,388,608 of control Run 3, whose K2a and K2b equal the judge's). With both
  changes, at 4,194,304 K2a is 0, 1 and 0 for s52, s60 and s42 (CoP split sd
  0.16, 0.07 and 0.03) where the standard panel has 12, 11 and 0; through
  10,485,760 the stance4 split under both stays at 0.03–0.21. At the finals it
  is K2a 0, 3 and 2 (split sd 0.20, 0.27 and 0.32, with 0, 2 and 1 panel
  falls). The controls with both changes have K2a 0 at every J but Run 3's
  8,388,608 (1). These panels change the task the judge rolls, so they
  diagnose; they are no fix and no verdict (D-D40 keeps them so, §20.3).

### 19.5 What the readouts found (report only)

Measured on the judge and panel files, the runs' evaluation ledgers and the
stored rollouts; "inferred" marks what is not measured.

1. **The stagger went, the split stayed (s52 and s60).** At 4,194,304 the feet
   were nearly in line: window stagger sd 4.8, 2.3 and 1.2 mm for s52, s60 and
   s42, against the zero-action statue's 2.5 mm and control Run 2's 11.9 mm at
   the same checkpoint. The per-foot centre-of-pressure split did not shrink:
   sd 0.99 and 1.09 in s52 and s60, against Run 2's 1.05 (s42 0.48, inside its
   saturation habit). Per mm of stagger that is 3.2, 7.3 and 6.4 times what
   the statue's flat-sole geometry gives; the controls are 0.5–1.4 times where
   their stagger was measured (3 of their 12 J checkpoints; at 4,194,304 only
   Run 2, 1.39). So in s52 and s60 the split moved from foot position to
   per-foot loading, one foot heel-loaded and the other toe-loaded, with no
   common lean (common CoP +0.07, −0.01 and +0.11 span half-lengths): their
   K2a failures there come from the split alone. Its drivers are the
   controls': the spawn's leg asymmetry and the prey's bearing (a two-factor
   fit explains R² 0.91, 0.89 and 0.90 of the stance4 split at 4,194,304 and
   0.86 of Run 2's). In s60 and s42 the policy over-corrects the spawn stagger
   (the window stagger's correlation with the spawn's is −0.56 and −0.31, the
   statue's +0.85), and the heel-loaded foot is still the forward one in 36
   and 26 of 40 episodes (s52 37). In the evaluation ledgers the split rose
   first (s52 from about 1.2M steps, s60 from about 2.5M) and the pad rail
   followed (first flags at 1.57M and 2.49M). (`results/split_form.md`,
   written at 06:01 UTC when only 4,194,304 had been judged.) Later the
   penalty held s60's feet in line (window stagger sd 1.8–5.5 mm to the
   final), but s52's stagger grew to 7.3–12.7 mm from 5,242,880 to 10,485,760
   (6.4 mm at the final), and s42's reached 19.5 mm at the final
   (`results/panels_summary.csv`); inferred, through that stretch part of
   s52's split was positional again. Inferred too, as the split-form analysis
   puts it: the stagger term removed the carrier it measures, not the split,
   and the next term should act on what remains, each foot's centre of
   pressure; the CoP-band prototype's band, [−0.6, +0.4] span half-lengths,
   has either foot's window mean outside it in 20, 22 and 11 of 40 stance4
   episodes at 4,194,304 (control Run 3 at 4,194,304: 0).
2. **The trajectory.** Every stagger run stood 39–40 of 40 clean at 1,048,576
   and had lost it by 2,097,152 (33, 37 and 5 of 40). At every judged
   checkpoint from 4,194,304 to 10,485,760, s52 and s60 read 21–30 of 40 (s52
   21–28, s60 27–30); s60's report-only checkpoints read 31 at 7,340,032, 39
   at 9,437,184 and 40 at 11,534,336. Both were back at 39 of 40 at the final;
   K1 counts only the judged checkpoints. s42 never recovered: 0–5 of 40 from
   2,097,152 to 9,437,184, then 17, 16 and 14.
3. **s42's actuator saturation.** From 1.83M steps s42's evaluations flagged
   the gate's actuator-saturation rail (`max_actuator_saturation_fraction`):
   70 of its 91 evaluations had at least 5 of 10 episodes on it, all from
   2,097,152 on, and the pre-registered alarm first fired at 2,228,224, in
   three spells (2,097,152–5,373,952, 26 evaluations; 5,636,096–10,616,832,
   39; 11,272,192–11,796,480, 5; `results/scores/`). On the panel the rail
   failed in 40, 37, 39, 37, 23 and 23 of 40 episodes at the six judged
   checkpoints (the judge files, session scratch only). At the final, besides
   those 23, 8 of 20 joint-space noise episodes fell (K3 allows 2) and 4
   full-horizon episodes broke the roll rail (K4 allows 3); the saturation
   rail is part of neither K3 nor K4. The s42 records give only the aggregate
   rail, not which actuator saturated: in stance3 Run 2 the same rail's habit
   was the tail actuator `tail_3_pitch_act` (§17.4), and whether s42's is the
   same actuator is not known. Applied afterwards, the same alarm fires on
   control Run 2 at 2,359,296 (44 of 91 evaluations; its criteria had no
   alarm) and not on Run 3 (1 of 91). The alarm is report only and was not
   acted on while s42 trained: it was first logged at the 04:36 UTC check-in,
   from evaluations up to 3,670,016 (`results/run_log.txt`). In stance3 Run 2
   it cost K1 (0, 0 and 36 of 40 clean at the first three judged checkpoints),
   after which that run passed every judged checkpoint, although the habit
   came back in its evaluations between them (§17.4); s42 never shook it.
   Which actuator saturated, when it began and what set it off are not
   diagnosed; an offline, report-only diagnosis is decided (D-D40's item 5a,
   §20.3) and has not run.
4. **s60's late backward shift.** s60's panel median moved back from
   10,485,760 on: `x_c` −17.4 and −20.9 mm from the statue at 10,485,760 and
   the final, against −5.8 to −12.6 mm at the four judged checkpoints before
   (−10.4 at 8,388,608; `results/checkpoints.csv`), with a common centre of
   pressure of −0.18 and −0.24 span half-lengths on the standard panel (−0.11
   to −0.01 before; `results/panels_summary.csv`), near control Run 3's −0.25
   at its 8,388,608 K2a failure (`results/split_form.md` §2). K2c passed
   (range 15.0 mm against 30), and through 10,485,760 its K2a failures still
   came with a split sd of 1.01–1.09 (`results/panels_summary.csv`). Inferred:
   a common fore-aft shift built up in s60 late, the kind the v6b ankle spring
   resists (D-D40's item 4a, §20.3).

### 19.6 Replicate study of the stagger penalty (scratch only, report only)

While the stance4 runs trained, the session ran a replicate of §17.5's and
§17.8's continuations: does the stagger term's continuation effect survive
more seeds? It was pre-registered in `night/spec.md`, written 00:10–00:35 UTC
on 2026-10-20, amended from 01:36 UTC after a review
(`night/review_handling.md`: eight findings, all accepted) and frozen at
01:48:58 UTC (sha256 `24af508a…` as frozen; its header's "01:50" was written a
minute before, as `night/review_handling.md` records), before the queue
launched at 01:49:08 UTC (`results/run_log.txt`) and so before any of its
training steps. Four new continuations ran from stance3 Run 3's Colab
8,388,608 checkpoint to E = 9,961,472, in the same local CPU harness as
§17.5's A and §17.8's local no-change run: two with the stagger penalty (ST2
and ST3, weight 0.30, tolerance 0.045 m) and two without (NC2 and NC3), on two
new training seeds, 152 (ST2, NC2) and 252 (ST3, NC3). With the three runs of
the night before that makes stagger n = 3 against no change n = 4. The primary
readouts at E are the window stagger sd and the K2a count, tested by an exact
one-sided permutation (rank-sum) test, 3 against 4. The reading lines, fixed
before launch: (1) complete separation on both readouts, the term's
continuation effect is real; (2) on one only, partial; (3) on neither, it does
not survive replication; a safety flag on a stagger run is named beside the
line ("met through a flagged run"). Measured before launch, the Colab
no-change run already had K2a 0, so K2a could not separate completely and line
(1) could not be reached; the spec kept the lines as fixed (its option (a),
decided at 01:48 UTC), so the strongest attainable reading was line (2). The
four new runs' judge files are relaxed-binding (report mode; seed and commit
mismatches allowed), and so are A's and the local no-change run's (report
mode; commit mismatch allowed); the Colab no-change run's is stance3 Run 3's
own judge file at E, fully bound (§17.5). The study is report only, so none of
its readings enters a stance4 verdict.

At E, on the 40-seed panel (measured; `night/readout.md`,
`night/readout.json`):

| Run | Arm, training seed | Clean of 40 | K2a | Window stagger sd (mm) | Safety flags |
|---|---|---|---|---|---|
| A (§17.5) | stagger, 52 | 40 | 0 | 7.5 | none |
| ST2 | stagger, 152 | 37 | 3 | 7.7 | 2 panel falls, a side door, reward ratio 0.945 |
| ST3 | stagger, 252 | 40 | 0 | 4.2 | none |
| local no-change (§17.8) | none, 52 | 25 | 15 | 25.9 | none |
| Colab no-change (§17.5) | none, 52 | 40 | 0 | 9.7 | none |
| NC2 | none, 152 | 37 | 3 | 14.8 | none |
| NC3 | none, 252 | 32 | 8 | 29.3 | clean below start for a new reason, 3 panel falls, a side door, reward ratio 0.942 |

- **The reading is line (2), partial, met through a flagged run (ST2).** The
  window stagger sd separates completely: every stagger run is below every
  no-change run (one-sided p 1/35 = 0.029). That p is optimistic, because
  three of the seven runs (A and the two no-change runs of §17.5 and §17.8)
  suggested the hypothesis and their order was known before the study: given
  that order it is 72/840 = 0.086 (3/35), and the new runs alone, 2 against 2,
  give 1/6 = 0.167. Within each seed the stagger run is the lower (−18.5, −7.1
  and −25.1 mm for seeds 52, 152 and 252; descriptive, no test). K2a does not
  separate: 0, 3 and 0 against 15, 0, 3 and 8 (p 7/35 = 0.200, or 192/672 =
  0.286 given the order). Each of the seven rollout sets reproduces its judge
  file exactly.
- **What it says, as read in the log (report only):** in continuation the term
  shrank the stagger it prices on every seed, and that did not carry over to
  the judged pad statistic, K2a. It does not bear on a from-scratch run: the
  stance4 screen (§19.2) is the deciding evidence for the term, and it read
  line 5.
- **Caveats.** Continuations of one parent, not runs from scratch; two new
  runs per arm; one flagged stagger run; a local CPU harness beside one Colab
  no-change run; relaxed-binding judge files for all but the Colab no-change
  run.

### 19.7 The CoP-band continuation test (scratch only, report only)

Line 5 names "the per-foot CoP band term (new contact-reward code)". The
session wrote it as a code-only prototype between 00:35 and 01:30 UTC on
2026-10-20 (`copband/design.md`; a proposal, nothing trained): an opt-in
`[env]` option of the T. rex env, off by default, on the scratch commit
`f94452ad` on `c1e146a8` (branch `scratch/cop-band`, never pushed). Each step
after the settle (step > 200), while the command is zero, it reads each foot's
whole-foot centre of pressure exactly as the judge's K2b statistic does, with
the recorder's own code, and charges a foot whose centre of pressure lies
outside [−0.6, +0.4] span half-lengths, or whose pad centre of pressure lies
outside [−0.65, +0.65], up to w/2 per step, as 1 − exp(−(u / 0.357)²) of it
for a distance u outside (tolerance 0.357, 25 mm), with w 0.30. Two guards
close side doors: a foot that is unloaded, carries under 35% of the load or
stands off its pad pays the full per-foot price. Off, it changes nothing
(measured on the prototype, byte for byte). After the diagnostic panels at
4,194,304 showed the per-foot loading split (§19.5 item 1), the session tested
it in continuation, pre-registered in `night2/spec.md` (frozen at 06:29 UTC,
sha256 `b1606f74…` as frozen, before its runs were launched at 06:52;
`night2/review_handling.md`: eight findings, all minor, no frozen file
changed). Two parents, s52 and s60 at their Colab 4,194,304 checkpoints, each
continued to E2 = 5,767,168 with its own training seed, with the band on (CB)
and with no change (NC): one run per arm per parent. The primary comparison is
CB minus NC within each parent, on K2a and on the CoP split sd. The reading
lines, fixed before launch: (1) CB lower on both readouts for both parents,
with no new safety flag: support for a CB arm in the next from-scratch screen;
(2) lower on both for one parent only, or on one readout for both: partial;
(3) otherwise, no support. A container reclaim at about 07:35 UTC cost about
80 minutes; CB52 and NC52 resumed from their 4,718,592 checkpoints. The judge
files are relaxed-binding (report mode; commit mismatch allowed), so none
enters a stance4 verdict.

At E2, on the 40-seed panel (measured; `night2/readout2.md`,
`night2/readout2.json`; the parents at 4,194,304 had K2a 12 and 11 and a split
sd of 0.99 and 1.09):

| Run | Clean of 40 | K2a | K2b | CoP split sd | Window stagger sd (mm) | Panel falls | Reward ratio |
|---|---|---|---|---|---|---|---|
| CB52 | 40 | 0 | 0 | 0.288 | 9.4 | 0 | 0.996 |
| NC52 | 22 | 18 | 18 | 1.203 | 6.7 | 0 | 0.994 |
| CB60 | 37 | 3 | 3 | 0.105 | 2.7 | 3 | 0.928 |
| NC60 | 18 | 22 | 21 | 1.272 | 6.1 | 0 | 0.981 |

- **The reading is line (2), partial**: CB is lower than NC on both readouts
  for both parents (K2a 0 against 18 and 3 against 22; split sd 0.29 against
  1.20 and 0.11 against 1.27), but through a run with new safety flags: CB60
  has 3 panel falls (hop-or-fall 3) and a reward ratio of 0.928, which NC60
  does not. The paired panel-seed bootstrap and exact McNemar tests favour CB
  within each pair (K2a −18, 95% interval −24 to −12, and −19, −25 to −13);
  they cover only which 40 spawns the panel draws, not the run-to-run variance
  of training.
- **The guards** (a guard rollout of the same panel with the band observing):
  in CB60, 242 of 59,836 window foot-steps (0.40%) were left uncounted, all in
  its 3 falling episodes, a foot unloaded (103) or under the 35% load share
  (139), and in 2 of the 3 a foot also on its digits alone (53 foot-steps); in
  CB52, none. Outside those falls no foot-step was lifted, lightened below 35%
  or carried on its digits, and the band runs' digit share stayed at
  0.20–0.22. The guards cannot see the route the design calls its main risk
  (`copband/design.md` §5, last row): each foot's centre of pressure held in
  band by asymmetric ankle torque while the feet stay staggered. In the s52
  pair the band run's window stagger sd was the higher, 9.4 against 6.7 mm
  (paired bootstrap +2.7 mm, interval +1.2 to +4.1), and its mean left-right
  ankle difference the larger, 0.60° against 0.08°; in the s60 pair both were
  lower (2.7 against 6.1 mm, 0.13° against 0.69°). That no foot found such a
  route is not claimed.
- **Cost.** The prototype adds 0.62 ms to an env step on one thread (1.40 to
  2.02 ms, +45%; `copband/design.md` §6.3); at launch CB52 trained at 272
  steps per second against NC52's 370 (`results/run_log.txt`).
- **Caveats.** One run per arm per parent; continuations from stance4's
  4,194,304 checkpoints, not runs from scratch; a local CPU harness;
  relaxed-binding judge files; the s60 pair's diagnostic panels were skipped
  at the deadline. At most it supports a band arm in a from-scratch screen,
  which D-D40 decides (§20.2).

### 19.8 What changes in §13–§18

- **§13 and §18, the stance4 screen:** judged; all three runs fail, and the
  decision table's line is 5, final. The stagger penalty is not adopted. No T.
  rex stance is certified, and the two-seed ask stands. Walking pilot 2 stays
  parked, its branch and criteria kept; the next round and its parent are
  decided in §20.
- **§14, this repository:** the folders `stance4/results/`, `stance4/night/`,
  `stance4/night2/` and `stance4/copband/`, with their sources and sha256 in
  `SOURCES.txt` and `SHA256SUMS` (now 344 files).
- **§14, branches:** unchanged; `claude/screen-trex-r9-armlt-stagger` at
  `c1e146a8`, never merged, with no pull request. The band prototype's branch
  `scratch/cop-band` (`f94452ad`) exists only in the session's scratch.
- **§14, Google Drive:** the three runs' folders `stance4_LTS_s52/`,
  `stance4_LTS_s60/` and `stance4_LTS_s42/` under
  `MyDrive/mesozoic-labs/logs/tyrannosaurus_rex/anatomical/ppo/` hold their
  checkpoints; each judged one's sha256 is in `results/checkpoints.csv`.
- **§14, session scratch only:** the stance4 judge and scorer
  (`scratchpad/foot/stance4/judge/`), the judge files
  (`scratchpad/foot/stance4/judged/`), the diagnostic panels' files
  (`scratchpad/foot/stance4/panels/out/`), the runs' fetched records and
  evaluation ledgers (`scratchpad/foot/stance4/runs/`), the split-form
  analysis's tables and scripts (`scratchpad/foot/stance4/analysis/`), the
  overnight runs with their judge files, rollouts, guard readouts and scripts
  (`scratchpad/foot/stance4/night/` and `scratchpad/foot/stance4/night2/`),
  and the band prototype's clone with its calibration tables and checks
  (`scratchpad/fix/scratch_copband/`, `scratchpad/foot/stance4/copband/`).

## 20. Status 2026-10-20, later (appended): decision D-D40, the CoP-band screen; D-D39 and D-D36 amended

*Appended 2026-10-20; §1–§18 above are unchanged, and §19 gains pointers to
this section (§19.2's last sentence and §19.5 item 3's last clause now say
what §20 decides).* After §19's outcome the maintainer took decisions in the
session's chat on 2026-10-20, after 15:30 UTC; the last lines of
`stance_2026_10/stance4/results/run_log.txt` (those beginning "MAINTAINER
DECISION") log them in short, numbered 1a, 2A, 3a–3d, 4a, 5a, 6 and 7a–7c
there. Where §20.2 records a detail that those lines leave out, it says where
the detail comes from. [BEHAVIOR_RECIPES_PLAN.md](../BEHAVIOR_RECIPES_PLAN.md)
§6.2 records them as decision D-D40, the next screen (1a to 6), and as dated
amendments of D-D39 (its outcome, §19, and its runtime, 7c) and of D-D36
(walking pilot 2's parent, 7a). Decision 7b asked for this record now,
documentation only. None of them trains anything on `main` or changes its
digests, and as screens and pilots the runs they start certify nothing.

### 20.1 D-D39 amended: the outcome, and the runtime confirmed

- **The outcome** (§19.2): all three runs fail (s52 and s60 fail K1 and K2,
  s42 fails K1–K4), H = 0 and K2p = 0, the credit rule fails, and the decision
  table's line is 5, final: the stagger penalty is not adopted. The best
  stagger final, s52's (39 of 40 clean, no noise falls), is one episode short
  of stance3 Run 2's final (40 of 40, no noise falls), which stays the parent
  the criteria offer walking pilot 2, and which D-D36's amendment makes pilot
  2's parent (§20.4). Of the two candidates line 5 names, D-D40 takes the band
  (§20.2).
- **The runtime.** The maintainer confirmed the Colab L4 GPU runtime for the
  stance4 screen (7c): all three runs recorded an L4 (the `gpu` block of each
  run's `stage_config.json`, measured; §19.1). This closes the confirmation
  §18.1 left pending. What §18.1 and §19.1 mark inferred stays inferred: that
  each run stayed on the L4 throughout (s60's and s42's `progress.json` record
  one session each; s52's was not fetched) and that the networks ran on the
  GPU. The L4 is also the band screen's runtime (7c).

### 20.2 D-D40: a from-scratch screen of the per-foot CoP band

- **The term (1a).** The maintainer approved a from-scratch screen of the
  per-foot centre-of-pressure band, line 5's "per-foot CoP band term (new
  contact-reward code)": a new reward term, an opt-in `[env]` option of the T.
  rex env, off by default, rather than the v6b ankle spring, line 5's other
  candidate. The term is §19.7's prototype
  (`stance_2026_10/stance4/copband/design.md`) with the three fixes and the
  weight of 3a–3d below.
- **The runs.** Three 12M runs from scratch on seeds 52, 60 and 42 on Colab L4
  GPU runtimes (7c), judged by criteria fixed before any training step, with
  K1–K4 unchanged. The seeds and the 12M are in the option text the maintainer
  locked in the chat, not in the log's 1a line, which keeps only the criteria
  (the note's §8 item 7 has "three seeds from scratch, 12M steps, Colab L4 as
  in stance4"). Physics r9 and `n_envs` 4 are in no decision's words; they
  follow from 2A (inferred): the arm is the stance4 recipe plus the band only,
  and stance4's criteria fix both.
- **The arm (2A): L+T + the stagger penalty (0.30, 0.045 m) + the band**, that
  is the stance4 recipe plus the band only, so the three stance4 runs (§19)
  are its control, one difference apart. The design note had recommended
  otherwise: its §8 item 5 recommended L+T plus the band alone (on
  `1af2d4d7`), so that the stance3 L+T runs stay a valid control, with a
  band-plus-stagger arm as the alternative if stance4 landed on line 4 rather
  than 5. The recommendation changed, and the maintainer was told so with
  the decision, because the overnight band test (§19.7) ran on stagger parents,
  and the stagger term prices the route the note calls its main risk (its §5,
  last row: each foot's centre of pressure held in band by asymmetric ankle
  torque while the feet stay staggered), which the band's guards cannot see.
  Inferred, not tested: without the stagger term nothing in the arm would
  price that route.
- **Three fixes to the prototype, and its weight (3a–3d), locked as
  recommended:**
  - 3a: guard 1's 35% load-share step becomes a ramp: the full price at a load
    share of 0.25 or less, no guard price at 0.35 or more, linear between. The
    note's §4.1 defines the step, and its §7 item 4 had named it a reward
    cliff that early training might sit near (inferred there); it never fired
    on the stored panels, and in §19.7's CB60 it fired only inside the 3
    falling episodes. Nothing recorded says that the step caused those falls.
  - 3b: a speed pass: one fused per-step reduction, pinned byte-identical to
    the recorder's readings, to cut the prototype's measured +0.62 ms per env
    step over band-off (1.40 to 2.02 ms on one thread; §19.7). No numeric
    target was decided: neither the log's 3b line nor the recommendation the
    maintainer locked gives one.
  - 3c: the contact reduction moves out of `environments/shared/gait/` into a
    neutral shared module, so that the gait layering rule (D-D23: no species
    env module imports the gait package) holds with no exception (the
    prototype imported the gait package lazily, and its layering test named
    that one exception); the note's recorder bit-identity test, which pins the
    band's readings to the recorder's (3b), moves with it (its §8 items 2 and
    3; the log's 3c line does not name the test).
  - 3d: the band weight is 0.30, the note's proposed dose, whose maximum
    equals the stagger term's (its alternative, 0.50, is not taken).
- **Not part of this change.** The band's code, criteria, judge and Colab
  steps are being built separately, on a scratch branch that is not pushed,
  and are not committed anywhere public; this records the decision only. The
  criteria are to be fixed before any training step, with K1–K4 unchanged and
  4a's trigger line in the decision table; their other contents are not
  recorded here. A later PR records them and the screen's outcome.

### 20.3 What else D-D40 settles

- **The v6b ankle spring is held in reserve (4a):** ankle stiffness 40 → 250
  N·m/rad, damping 45 → 90 and the spring reference at the statue's settled
  78.6°, a plant revision. The band screen's fixed decision table gets a
  trigger line: if the band fixes the split but runs fail K2 on drift (a
  common fore-aft shift), the next round tests the v6b spring. Inferred: the
  spring resists mainly a common ankle opening, which is not what failed at
  4,194,304 (§19.5 item 1), and s60's late backward shift (§19.5 item 4) is
  the kind of failure it addresses.
- **Actuator saturation (5a): no training change.** During the band screen the
  session reads each run's evaluation ledger at every check-in and tells the
  maintainer at once when the saturation alarm fires (run, step, actuator). In
  parallel, an offline, report-only diagnosis of s42's saturation (which
  actuator, its onset, its trigger; §19.5 item 3), started after the band
  build's timing measurement so that it does not load the machine during it.
  Fails stay fails, and no run is stopped or restarted on an alarm.
- **Diagnostic panels (6):** mirrored-leg spawns and the prey straight ahead
  stay diagnostic panels only, adopted into neither training nor the criteria;
  the band screen runs them at every scored checkpoint, report only, as a
  yardstick (§19.4).
- **Still open:** the signed per-foot bar in the gate (§16.1 item 5), and
  whether a screen run could ever count toward `certification_seeds` (§16.2).

### 20.4 D-D36 amended: walking pilot 2's parent (7a, option i)

Walking pilot 2 will be rebuilt for stance3 Run 2's final
(`stance3_armLT_s60`'s `stage1_final`, sha256 `cc21604a…`: 40 of 40 clean, no
noise falls; §17.2) as its parent, the parent that D-D39's comparison keeps
(§19.2). Pilot 2's criteria (§15.7) still name r9 arm L seed 60 at 6,291,456
steps (`70f3a444…`), so they are amended before any pilot-2 training step, the
new parent is committed on pilot 2's branch, and its start-up checks are
re-run. The rebuild comes right after the band screen's package is ready, and
the maintainer starts the pilot when a fourth Colab slot is free. Until then
pilot 2 stays parked and untrained. It is still a deliberate exception to GQ-3
(a) and GQ-17 of
[GAIT_QUALITY_PLAN_2026_09.md](../GAIT_QUALITY_PLAN_2026_09.md), both still
open, and it certifies nothing. The amended criteria and the rebuilt branch
are not part of this change; a later PR records them and the pilot's outcome.

### 20.5 What changes in §13–§19

- **§13, §18 and §19:** the next round is decided (D-D40, a from-scratch
  screen of the band on the stagger arm); it is not yet built in public and
  not trained. No T. rex stance is certified, and the two-seed ask stands.
- **§13, §17.6 item 2, §18.2, §19.2 and §19.8, walking pilot 2:** its parent
  is decided, stance3 Run 2's final; its criteria are to be amended and its
  branch rebuilt, and it stays parked until then.
- **§17.6 item 3 and §19.5 item 3:** the saturation alarm is read at every
  check-in of the band screen and reported at once; s42's saturation gets an
  offline diagnosis, report only, not yet run.
- **§18.1, the runtime:** confirmed, the L4.
- **§14:** no folder, file or branch is added to this repository by §20; the
  band's package and pilot 2's rebuild are outside this change. Pilot 2's
  branch `claude/pilot-walk-r9-b` (`ba464855`) is unchanged.

## 21. Status 2026-10-22 (appended): the stance5 band screen judged

*Appended 2026-10-22; §1–§20 above are unchanged, so §19's and §20's "never
pushed", "being built", "not committed anywhere public", "not yet run" and "not
yet trained" are the wording of 2026-10-20.* The per-foot centre-of-pressure
band of D-D40 (§20.2) was built and packaged on 2026-10-20 and 2026-10-21; its
three screen runs trained to their final 12,001,280 steps on Colab L4 GPU
runtimes from 2026-10-21 to 2026-10-22 and were judged by their fixed criteria
with the stance5 judge and scorer, the verdict of record written at 04:57 UTC on
2026-10-22 (the score files' time; the run log's entry is timed 04:58). This
section records the build and the package (§21.1), the runs and
the judging (§21.2), the verdicts, the decision-table line and the parent
comparison, which are results (§21.3), and the report-only extras, gates,
certificates, panels, band readouts and saturation watch (§21.4); and, from the
same days, the offline diagnosis of stance4 s42's saturation that D-D40's 5a
asked for (§21.5) and walking pilot 2's rebuild under D-D36's amendment of
2026-10-20 (§21.6). The maintainer's decisions that followed, on 2026-10-22, are
recorded in §22. A screen certifies nothing: the verdicts are the criteria's,
not a gate's, nothing here certifies a run, writes a result bundle or promotes
anything, and `main` and its digests are unchanged. The evidence is in the
folder `stance5/` of [stance_2026_10/](stance_2026_10/README.md), whose
`results/` has its own [README](stance_2026_10/stance5/results/README.md), and
in `stance4/satdiag/` and three files of `pilot2/` of the same folder, all added
on 2026-10-22; paths in backticks below are relative to `stance5/` unless they
start with `../`, and every number below is read from `results/run_log.txt`, the
score files in `results/scores/` or the files named. A fact-check of the
session's human-facing report on these runs (a document outside the repository,
not copied) checked ten statements and fixed them there (`results/run_log.txt`,
05:15 UTC); this section uses the corrected forms.

### 21.1 The band build and the package

- **The build (D-D40 3a–3c).** On the scratch branch `scratch/cop-band-v2`, on
  the prototype `f94452ad` (itself on `c1e146a8`), all on 2026-10-20: 3c, the
  contact reduction moved out of `environments/shared/gait/` into the neutral
  package `environments/shared/floor_contact` (`3a1d5de3`); 3b, one fused
  per-step reading (`4819eb6c`); and 3a, the load-share guard as a ramp
  (`a15494ed`). Then the review's fixes (`99e205bb`, 00:59 UTC on 2026-10-21): a
  float reward path, a foot counted only at a guard of 1, no gait or curriculum
  module loaded at run time, and no 1e-8 floor. A container restart at about
  22:38 UTC on 2026-10-20 lost the first fix step, which was re-run. The build's
  final check, at about 01:30 UTC on 2026-10-21: band off byte-identical to
  `c1e146a8` (10 of 10; tasks `3fa46708…` and `d4a3b8f8…`), the recorder
  unchanged, band on equal to the recorder's readings on 16,000 foot-steps with
  no mismatch, the full test suite with "the known 47 failures only", the digest
  current and ruff clean.
- **The timing.** The final band-on overhead is +0.245 to +0.378 ms per env step
  (+18–30%; an interim measurement at 21:10 UTC on 2026-10-20 gave +0.27 to
  +0.34 ms, +19–21%), against the prototype's +0.62 ms (§19.7). The build's own
  goal of 0.15 ms, which no decision set (§20.2, 3b), was not met.
- **3d, the config commit `cf80ecb1`** (authored at 02:38 UTC and committed at
  05:41 UTC on 2026-10-21, after the criteria's review; its message cites their
  sha256): `stance_cop_band_weight` 0.30 in the T. rex stance configuration,
  every other band key at its default, and the collapse reference 4457.4
  (stance4's method: production's 3756.8 plus the band arm's statue gain,
  700.55). The band-on stance task is `sha256:8c6998d2…`. Measured on the
  zero-action statue (`criteria.txt`): the band costs it 1.91 per full-horizon
  episode and changes no dynamics; off, it leaves stance4's task at `3fa46708…`.
- **The package**, done at about 07:15 UTC on 2026-10-21: `criteria.txt` (sha256
  `3b28963f…`, re-fixed after its review and before any training step; the first
  text was `c2809201…`), with K1–K4 as stance4's and stance3's, the three
  stance4 runs as the control, not retrained, the credit rule, a drift-or-split
  rule for D-D40 4a's trigger line, and a decision table of six lines;
  `criteria_notes.md`, which maps each decision of 2026-10-20 to the criteria's
  lines; the judge `judge/judge_stance5.py` (`d9286724…`) and the scorer
  `judge/score_stance5.py` (`d5515041…`; 101 test cases pass, and it reproduces
  stance4's verdicts); the Colab steps (`colab_steps.txt`); and smoke evidence,
  110 of 110 checks.
- **The screen branch was pushed.** `claude/screen-trex-r9-armlts-copband` was
  on GitHub at `cf80ecb1` by about 07:15 UTC on 2026-10-21 (checked with `git
  ls-remote`), never merged, with no pull request; its history holds `c1e146a8`,
  the prototype `f94452ad`, the four build commits and `cf80ecb1`. This
  supersedes, for the commit `f94452ad` and the band's code, §19.7's and §20.2's
  "never pushed", "a scratch branch that is not pushed" and "not committed
  anywhere public", the wording of 2026-10-20, left unchanged above; the scratch
  branches themselves were not pushed (§21.7).
- **A slip.** At about 07:17 UTC two of the package's files were copied into the
  session's checkout of the repository, against the standing rule; both were
  untracked new files, they were removed within a minute, and the checkout's
  `git status` was clean afterwards.

### 21.2 The runs and the judging

- **The runs.** By their `provenance.json` files (as the run log records them at
  13:36 UTC on 2026-10-21), `stance5_LTSB_s60` started at 13:32:45,
  `stance5_LTSB_s52` at 13:33:09 and `stance5_LTSB_s42` at 13:34:32 UTC, each on
  commit `cf80ecb1` (not dirty), its seed, `n_envs` 4, r9 and MuJoCo 3.10.0. A
  first s52 start at 13:26 UTC carried its run label as its run id; the
  maintainer stopped it and restarted under the right id, as the session asked.
  At 13:46 UTC each run's `01_stance/stage_config.json` read the commit, the
  seed, `n_envs` 4, the band weight 0.3 and guard shares 0.25 and 0.35, the
  stagger weight 0.3, the collapse reference 4457.4 and the task `8c6998d2…`,
  and its `progress.json` a zero-action reward of 4367.8, equal to the local
  measurement.
- **The runtime, measured.** The `gpu` block of each `stage_config.json` reads
  an NVIDIA L4 (23.7 GB, CUDA 13.0). Each run's `progress.json` records one
  session, started (`session_started_at`) at 13:37:07 (s52), 13:36:43 (s60) and
  13:38:28 UTC (s42) and ended at 12,001,280 steps at 03:32:11, 04:03:00 and
  03:14:04 UTC on 2026-10-22, at 239.5, 230.9 and 245.2 steps per second, with
  no warnings, so that each ran on the L4 throughout is inferred.
- **Fetch and binding.** As in §19.1: each judged checkpoint was fetched
  read-only from Drive, and each VecNormalize file transcribed twice and the
  copies compared, with a third transcription where they differed (s52 at
  7,340,032 and s60 at 10,485,760). Three place steps were cut off by a safety
  classifier and done by hand with the workflow's own command (s60 at 8,388,608;
  s52 and s60 at 10,485,760). The judge bound the plant, the seed, the task, the
  symmetry map and the run's `stage_config.json`; in all 36 judge files every
  binding matched, with no hard failure and no relaxation
  (`results/checkpoints_csv.py` asserts both for every file). The judge and
  panel queues, launched at 19:40 UTC on 2026-10-21, failed to start and ran
  from 20:31; they were stopped by container restarts at 21:30 UTC and about
  04:21 UTC and by a suspension of the idle container after 22:11 UTC (its cause
  inferred), and each interrupted job was re-run.
- **The judged checkpoints (J)**: 4,194,304 (judged from 20:37 to 20:48 UTC on
  2026-10-21), 5,242,880, 6,291,456, 8,388,608, 10,485,760 and the finals
  (judged at 04:05, 04:28 and 04:55 UTC on 2026-10-22 for s42, s52 and s60; the
  judge files' times), with 7,340,032 judged as a report-only checkpoint. The
  other report-only 1,048,576-step checkpoints (1,048,576, 2,097,152, 3,145,728,
  9,437,184 and 11,534,336) were never fetched, so 15 queued judge jobs found
  nothing to judge, and the late-stable S3 was not judged.
- **The score.** `results/scores/score_final.txt` and `.json`, written at 04:57
  UTC on 2026-10-22 (the files' time; the run log's entry is timed 04:58) by
  `score_stance5.py --runs-ended`, the verdict of record:
  it loads stance3's frozen scorer unchanged
  (`../stance3/judge/score_stance3.py`, sha256 `2c3879d3…`), extends only its
  judge-identity constants for the band arm and the control, checks the
  criteria's sha256 (matched) and stance4's `score_final.json` (`7c98d8ae…`,
  matched), and re-scores the control from its judge files as a cross-check.
  Re-run on the judge files, it reproduces the record apart from the path it
  names and s60's `best` and `robust_best` cells, which were judged after it
  (05:01 and 05:06 UTC; the judge files' times). `results/checkpoints.csv` has
  one row per judge file (36), recomputed with the frozen scorer's own counts
  (25 cells agree with the score).

### 21.3 The verdicts: Run 1 holds; line 3

Per run, the judged checkpoints are 4,194,304, 5,242,880, 6,291,456, 8,388,608,
10,485,760 and the final (4.2M to 12M below); the offsets are the panel median
of the both-feet whole-foot centre of pressure, `x_c`, against the statue's 64.0
mm ahead of the MTP III joints on the same seeds.

- **Run 1, `stance5_LTSB_s52` (seed 52): holds.** Clean 40 of 40 at all six,
  `stance_quality/v2` PASS throughout (LCB 0.928). K2a 0 and K2b 0 at every
  judged checkpoint; K2c passes (offsets −4.6, +2.8, −0.2, +0.2, −14.2 and −14.7
  mm, a range of 17.4 mm). K3 passes (0 of 20 joint-space and 0 of 20 own-sigma
  noise falls at 6,291,456 and at the final) and K4 passes (roll rail 0 and 0 of
  40). It holds fore-aft.
- **Run 2, `stance5_LTSB_s60` (seed 60): fails K3 only.** Clean 39, 40, 40, 40,
  40 and 39 of 40, v2 PASS at all six; K2a 0 and K2b 0 throughout; K2c passes
  (offsets +6.5, −15.0, −1.8, +6.3, +2.8 and −5.7 mm, a range of 21.6 mm). **K3
  fails**: 3 of 20 joint-space noise falls at the final (bar 2; 0 at 6,291,456;
  own-sigma 0 and 0). K4 passes (0 and 0). It holds fore-aft.
- **Run 3, `stance5_LTSB_s42` (seed 42): fails K1 and K3.** Clean 16, 27, 18,
  15, 38 and 19 of 40; in most failing panel episodes the gate's
  actuator-saturation rail failed (§21.4). **K1 fails**: v2 FAIL at 6,291,456
  and at the final, and five judged checkpoints below 33 of 40. K2 passes: K2a
  and K2b 0 at every judged checkpoint but the final, where each counts 1, an
  episode that fell (the rule counts a fall as over), and a K2c range of 18.3
  mm. **K3 fails**: 13 of 20 joint-space noise falls at the final (0 at
  6,291,456), and 1 of 20 own-sigma falls, the limit. K4 passes (0 and 0).

Per K test, against the control (the three stance4 runs, as recorded in
stance4's score of record, §19.2; the scorer reproduces their verdicts from
their judge files):

| Test | s52 | s60 | s42 | Band runs passing | Control passing |
|---|---|---|---|---|---|
| K1, the gate held | PASS | PASS | FAIL | 2 of 3 | 0 of 3 |
| K2, fore-aft held | PASS | PASS | PASS | 3 of 3 | 0 of 3 |
| K3, robustness | PASS | FAIL | FAIL | 1 of 3 | 2 of 3 |
| K4, roll | PASS | PASS | PASS | 3 of 3 | 2 of 3 |

**The recipe verdicts: one run holds; H = 1 and K2p = 3.** Fore-aft holds on s52
and s60. The credit rule fails on both parts: (a) one band run holds where two
are needed, and (b) the band arm passes K3 less often than the control (1 of 3
against 2 of 3; the rule's general and specific forms agree). No band run fails
K2, so none is classed drift or split (D = 0, S = 0). **The decision table's
line is 3, final**: "H == 1: not adopted; not separated from seed luck;
replicate on 2 more seeds before any commit." The band is not adopted. What
comes next is the maintainer's decision: D-D41 (§22).

**The parent comparison** (report only; stance3's rule: most clean panel
episodes at the final, ties to fewer joint-space plus own-sigma noise falls):
**the best band final, s52's** (`stage1_final`, sha256 `24a4fada…`: 40 of 40, 0
noise falls), **ties stance3 Run 2's final** (`cc21604a…`: 40 of 40, 0), and the
rule names both and leaves the choice to the maintainer. Behind them: the best
stance4 final (s52's, 39 of 40, 0), s60's band final (39 of 40, 3) and s42's (19
of 40, 14). Line 1's stance parent does not arise. The comparison does not
choose walking pilot 2's parent, which stays stance3 Run 2's final (D-D36 as
amended on 2026-10-20; §22).

### 21.4 Report only: the extras, gates, certificates, panels and readouts

None of these enters a criterion (`results/checkpoints.csv`, `results/scores/`,
`results/extras.txt`, `results/panels_summary.csv`). The fresh, tail and
common-noise numbers below were measured for this record from the judge files
(`results/extras.py`); the run log gives none.

- **7,340,032, judged like the J checkpoints:** s52 39 of 40 (its one failing
  episode on the saturation rail), s60 40 of 40, s42 19 of 40 (9 of 20
  joint-space noise falls).
- **Selection (S0 and S1):** in each run `robust_best` and `best` are one
  checkpoint: 917,504 steps in s52 and s42, both 40 of 40 with no own-noise
  falls and K2a and K2b 0; 6,815,744 in s60, 40 of 40 with 2 of 20 joint-space
  falls (judged after the score was written). The finals are as in §21.3.
- **The fresh panel 7042–7081 at the final:** s52 FAIL 38 of 40 (1 fall, a
  nosedive, and 2 hop-or-fall episodes against the gate's 1), s60 PASS 38 of 40
  (0 falls), s42 FAIL 16 of 40 (2 falls); the criteria's references are the
  stance4 finals' 39, 38 and 14 of 40 and stance3 Run 2's final's 40 of 40.
- **The forward-start tail panel (32 seeds) at the final:** 3, 0 and 4 falls for
  s52, s60 and s42; the statue 0, stance3 Run 2's final 0 and the stance4 finals
  2, 6 and 5.
- **The common-noise panel (40 seeds) at the final:** 6, 5 and 31 falls; the
  stance4 finals fell 5, 1 and 30.
- **The pre-registered readouts:** K5 flags no run. The late-fall fraction is
  above 5% only in the first 262,144-step bin (50.2%, 48.9% and 46.3%) and in
  s42's bin at 11,796,480 (6.6%). The run log records no result for the
  20-episode quick panels at every 524,288-step checkpoint, the per-term reward
  decomposition or the stagger watches, and this record has none.
- **The notebook's own end-of-stage gate**, `stance_quality_heading/v1`, on each
  run's `robust_best` (`results/notebook_gates/`): all three PASSED, 40 of 40
  seeds clean at six headings (LCB 0.928), at 917,504 steps (s52, s42) and
  6,815,744 (s60). It judged an early checkpoint, not the finals the criteria
  judge.
- **The stage's six-heading certificate rolled offline on the s52 and s60
  finals** (`results/certificates/`; stance3's runner, in a clone of the screen
  branch at `cf80ecb1`, on the band-on task `8c6998d2…`): s52 PASSED, 40 of 40
  seeds clean at all six headings (LCB 0.928, 988 s); s60 PASSED, 39 of 40
  (0.887, 1,103 s). Neither is gate evidence: the band is not on `main`.
- **Diagnostic panels (D-D40 6)** at every judged checkpoint: at the finals,
  clean of 40 under the standard panel, mirrored legs, the prey straight ahead
  and both, s52 40, 40, 40 and 40, s60 39, 37, 39 and 37, s42 19, 8, 13 and 7,
  with falls 0, 0, 0 and 0 (s52), 0, 0, 0 and 0 (s60) and 1, 0, 1 and 0 (s42).
- **The band readouts.** At every judged checkpoint no full-horizon episode of a
  band run had either foot's window-mean whole-foot centre of pressure outside
  the band [−0.6, +0.4] (0 of 40, s42's final 0 of 39), except one of s60's at
  8,388,608; the control's judge files, read the same way, give 10 to 28 of 40
  per judged checkpoint for stance4 s52 and s60 and 0 to 21 for s42
  (`results/scores/`). Per foot-step the band was left: up to 1,012 of 64,000
  window foot-steps in a standard panel (s60 at 8,388,608). The guard stood at 1
  on every window foot-step of the full-horizon episodes of every panel, never 0
  and never on its ramp, so the lighten door (§20.2, 3a) was not used there.
- **The twist-route watch** (read for this record from the two panels summaries,
  the criteria's reading, which is inferred from the statics): the band runs'
  per-foot centre-of-pressure split sd is below the same seed's stance4 run's at
  every judged checkpoint (0.03 to 0.40 span half-lengths, against 0.30 to
  1.25), while their window stagger sd is above it at 9 of the 18 (s42 at the
  first five, s60 at 4.2M, 5.2M, 8.4M and the final; by 0.1 to 5.5 mm, at most
  8.0 mm, against the statue's 2.5). Their ankle servo-error asymmetry is 0.05
  to 0.81 deg, against the control's 0.56 to 2.14 deg (`criteria.txt`). Whether
  any of it is the route the design note feared is not tested.
- **The saturation watch (D-D40 5a)** (`results/saturation/`,
  `results/scores/`). s42: its ledger's first episode on the saturation rail at
  3,145,728 (1 of 10); the first flagged evaluation (5 or more of 10) at
  3,538,944; the alarm (two flagged in a row) at 3,670,016, against stance4
  s42's 2,228,224. It was logged at 19:05 UTC on 2026-10-21 (a check-in, from
  the evaluations in `progress.json`; that the maintainer was told then, as 5a
  asks, is inferred), with its step from the ledger at
  19:40 and its actuator at 19:48. In all, 30 of s42's 91 evaluations were
  flagged: 26 in four alarm runs (3,538,944–4,718,592, 10; 6,160,384–6,553,600,
  4; 6,815,744–7,208,960, 4; 7,864,320–8,781,824, 8) and 4 alone (5,111,808,
  5,898,240, 10,878,976 and 11,927,552). Rollouts of the 4,194,304 checkpoint
  named **`tail_3_pitch_act` at +1 (tail raised)**: its command at 0.9 or more
  on 77% of window steps (mean +0.92), and 10 of 20 episodes over the gate's
  bar; at 5,242,880, 67% (+0.92) and 6 of 20; at 6,291,456, 75% (+0.90) and 14
  of 20. That is the actuator and sign of stance4 s42 and stance3 Run 2 (§21.5).
  On the judged panels the rail failed in 24, 13, 22 and 24 of 40 episodes at
  4.2M to 8.4M, 2 at 10.5M and 20 at the final (the run log's 04:24 UTC "21" is
  corrected to 20 at 05:15). s52 and s60 had no flagged evaluation (0 of 91
  each). No run was stopped or changed on the alarm.

### 21.5 Report only: the stance4 s42 saturation diagnosis (D-D40 5a)

The offline diagnosis of `stance4_LTS_s42`'s actuator saturation (§19.5 item 3)
ran from 2026-10-20 (after the band build's timing measurement) to 2026-10-21,
02:40 UTC; its report and tables are in `../stance4/satdiag/`. Deterministic
rollouts of s42's checkpoints, with stance4 s52's and s60's and stance3 Run 2's
beside them, through the judge's own loading path; the five runs' eval ledgers;
a counterfactual; and eight verifiers. Measured:

- **The actuator:** `tail_3_pitch_act`, the distal tail pitch servo, commanded
  at +1 (tail tip up), the actuator and sign of stance3 Run 2. The command is
  pinned, not the joint or the motor: at 4,194,304 the joint sits 0.216 rad
  below its stop on 0.26 of its force limit, so the ledger's force columns
  cannot see it; the eval ledger has no per-actuator column, and
  `gate_progress.npz`'s `action_dc_per_actuator` could name it.
- **The onset:** between the placed checkpoints 1,048,576 (clean) and 2,097,152
  (17 of 20 rolled episodes flagged); the ledger flags from 1,835,008 and the
  alarm comes at 2,228,224. Within an episode the command ramps up from the
  settled pose.
- **Its cost:** holding tail_3 at 0 in the s42 policy gains +85.9 (at 2,097,152)
  and +73.1 (at the final) per 800-step window, deterministically, with no
  falls; why training keeps the habit is not measured.
- **s52 and s60** put no actuator at 0.9 or more in the window at 2,097,152,
  4,194,304 or the final (one s52 episode at 8,388,608 held `neck_pitch_act`
  there for 702 window steps); their tail_3 ended tail-down.
- **Candidate fixes** for a next round, none acted on: T1 and T2 (telemetry),
  P1, R1, R2 and Y1 (a policy-mean bound, reward changes and a plant change,
  each needing approval) and N0 (no training change, the current 5a).

Stance5's s42 took the same habit (§21.4): three of the eight runs so far on
L+T-based recipes have raised a saturation alarm (stance3 Run 2, stance4 s42 and
stance5 s42), all on this actuator and sign.

### 21.6 Walking pilot 2 rebuilt for its new parent (D-D36, as amended on 2026-10-20)

Started at 17:00 UTC on 2026-10-21, after the band screen's package (§21.1), and
done at 19:25 UTC (`../pilot2/rebuild_1021_summary.md`):

- **The parent** is stance3 Run 2's final (`stance3_armLT_s60`, 12,001,280
  steps; `stage1_final.zip` `cc21604a…` and its normaliser `894e86f4…`), with
  the plant (r9) and the policy architecture of pilot 1's parent; its action std
  is 0.064 on average, 0.009 at the knees and 0.125 at most (pilot 1's parent:
  0.095, 0.018 and 0.164).
- **The branch** `claude/pilot-walk-r9-b` gained three pilot-only commits on
  `ba464855`: `6a13a303` (the new parent and its `SOURCE.txt`), `8fc76eec` (a
  closing note in pilot 1's parent's `SOURCE.txt`) and `8c5478a6` (two sentences
  of the new `SOURCE.txt` corrected after review); only `pilot_checkpoints/`
  changes.
- **The criteria were amended on 2026-10-21, before any pilot-2 training step**
  (`../pilot2/criteria_2026-10-21.txt`; the pre-amendment text is
  `../pilot2/criteria.txt`, sha256 `a240ff5d…`): the parent, the branch head,
  the parent's numbers, the readings reworded (pilot 2 now differs from pilot 1
  in the parent and in the noise reset, so no outcome is attributed to either
  alone), a hold on the branch, and the pre-launch checks. No outcome
  definition, threshold, guard, label, seed, budget or tool changed.
- **The pre-launch checks**, on `8c5478a6`: the notebook smoke (its "Warm start"
  line with the new parent's std, the stage configuration's commit and parent),
  the judge accepting the run and refusing pilot 1's parent, and 29 warm-start
  tests; on `8fc76eec`, the full suites (5,937 passed).
- **The push.** The fast-forward from `ba464855` to `8c5478a6` was first refused
  by the session's permission classifier (19:25 UTC), was not worked around, and
  was made at 19:49 UTC after the maintainer's "Go ahead and push it" (checked
  with `git ls-remote` before and after). From then on nothing more is pushed to
  the branch until all three pilot-2 seeds are judged. The Colab steps record
  the push (`../pilot2/colab_steps_2026-10-21.txt`).

Pilot 2 is not yet launched or trained, and has no result. It is still a
deliberate exception to GQ-3 (a) and GQ-17 of
[GAIT_QUALITY_PLAN_2026_09.md](../GAIT_QUALITY_PLAN_2026_09.md), both still
open, and it certifies nothing. Its parent stays as it is for now (§22).

### 21.7 What changes in §13–§20

- **§13 and §20.2, the band screen:** judged; one run holds, and the decision
  table's line is 3, final: the band is not adopted, and the line asks for two
  more seeds before any commit (D-D41, §22). No T. rex stance is certified, and
  the two-seed ask stands.
- **§19.7, §19.8 and §20.2, the band's code, criteria, judge and Colab steps:**
  the code is on the pushed screen branch, and the rest is in `stance5/` of the
  evidence folder.
- **§20.3, saturation (5a):** the ledgers were read at every check-in from 19:40
  UTC on 2026-10-21, when they were first placed (before that, the latest
  evaluations in `progress.json`); the alarm fired on s42 and named
  `tail_3_pitch_act` at +1 (§21.4); stance4 s42's offline diagnosis is done,
  report only (§21.5).
- **§20.4, walking pilot 2:** rebuilt, its criteria amended and its branch
  pushed at `8c5478a6`; not yet trained (§21.6).
- **§14, this repository:** the folders `stance5/` and `stance5/results/`,
  `stance4/satdiag/`, and three files in `pilot2/`, with their sources and
  sha256 in `SOURCES.txt` and `SHA256SUMS` (now 381 files).
- **§14, branches:** `claude/screen-trex-r9-armlts-copband` at `cf80ecb1` and
  `claude/pilot-walk-r9-b` at `8c5478a6` are on GitHub, never merged, with no
  pull request. The band's scratch branches `scratch/cop-band` and
  `scratch/cop-band-v2` exist only in the session's scratch.
- **§14, Google Drive:** the three runs' folders `stance5_LTSB_s52/`,
  `stance5_LTSB_s60/` and `stance5_LTSB_s42/` under
  `MyDrive/mesozoic-labs/logs/tyrannosaurus_rex/anatomical/ppo/` hold their
  checkpoints; each judged one's sha256 is in `results/checkpoints.csv`.
- **§14, session scratch only:** the judge files
  (`scratchpad/foot/stance5/judged/`), the diagnostic panels' files
  (`scratchpad/foot/stance5/panels/out/`), the certificates' per-episode files
  (`scratchpad/foot/stance5/cert/`), the runs' fetched records and evaluation
  ledgers (`scratchpad/foot/stance5/runs/`), the build's records
  (`scratchpad/foot/stance5/build/` and `scratchpad/foot/stance4/bandv2/`), the
  satdiag rollouts and verifiers (`scratchpad/foot/stance4/satdiag/`) and pilot
  2's rebuild evidence (`scratchpad/foot/pilot2/rebuild_1021/`).

## 22. Status 2026-10-22, later (appended): decision D-D41, the band's replication; D-D40 and D-D36 amended

*Appended 2026-10-22; §1–§21 above are unchanged.* After §21's outcome the
maintainer took decisions in the session's chat on 2026-10-22; the run log
(`stance_2026_10/stance5/results/run_log.txt`) logs them in short in its entries
timed 15:08, 15:11 and 16:32 UTC (those with "MAINTAINER DECISION"), and an OK
for two report-only workflows at 15:52 UTC; these are its times as corrected the
same day (the entries first carried guessed times;
`stance_2026_10/stance5/results/README.md`). Where §22.1 gives a detail those
entries leave out, it says so.
[BEHAVIOR_RECIPES_PLAN.md](../BEHAVIOR_RECIPES_PLAN.md) §6.2 records them as
decision D-D41, the replication, and as dated amendments of D-D40 (its build,
runtime and outcome, §21.1–§21.5) and of D-D36 (walking pilot 2 rebuilt, §21.6).
Decision (4) asked for this record now, documentation only. None of them trains
anything on `main` or changes its digests, and as screens and pilots the runs
they start certify nothing.

### 22.1 D-D41: the band's replication

- **(1) Replicate the unchanged band recipe** (15:08): arm LTSB from the screen
  branch at `cf80ecb1` (§21.1).
- **(2) Two band runs only, seeds 70 and 80** (15:08), judged on K1–K4 against
  the existing stance4 control, which is not retrained, with new criteria fixed
  before training (the log names "noise level for the new seeds, how the 5 band
  runs are read"). 12M steps, physics r9, `n_envs` 4 and a Colab L4 runtime are
  in no decision's words; they follow from "unchanged" (inferred).
- **The reading rule** (15:11): ADOPT only if both new seeds hold (H = 3 of 5);
  the K3 pass fraction against the control (3 of 5 against 2 of 3 at best) is
  reported, not a veto. The log gives the reason in parentheses: under stance5's
  rule (b) adoption would be impossible, since the band arm already has two K3
  failures (§21.3).
- **The new seeds' joint-space noise sigma.** The K3 test reads a per-seed
  `control_sigma`, which the frozen parameters hold only for seeds 42, 52 and 60
  (the 05:15 UTC entry). The new seeds' is the element-wise mean of those three:
  at 15:11 it was the session's default, stated to the maintainer with the
  reading rule ("my default, stated to the maintainer", the log says), not a
  choice the maintainer stated; in the entry timed 16:32 the maintainer
  confirmed it, one vector for both new seeds.
- **(3) No parent change now** (15:08): walking pilot 2 stays on stance3 Run 2's
  final, and s52's final (which ties it, §21.3) is reconsidered after the
  replication.
- **(4) The stance5 records PR now, documentation only** (15:08): this change.
- **Run handling** (the entry timed 16:32): defaults adopted for the
  replication. A run cut off before its first periodic checkpoint and restarted
  under `<RUN_ID>_b` is that run; a judge refusal on the run's own records
  counts as an invalid J file; a resumed run is judged as it stands, with
  superseded files moved aside, never deleted; a cut-short final resumes from
  the newest intact periodic checkpoint in the same run id, and the final slot
  is only a `stage1_final` pair at 12,001,280; an early stop makes the final J
  file invalid; "ended" and "exists" are defined as proposed; on a refusal to
  reproduce stance5 nothing is read until the cause is found; and with no L4 on
  a resume the maintainer is asked first.
- **The package** (the entry timed 16:32): built and validated. Its judge
  reproduces stance5's s52 judge file at 4,194,304 field for field, its scorer
  reproduces stance5's three verdicts in a validation run, 37 tests pass, and
  its criteria were fixed at sha256 `10464eab…`; they are being re-fixed for the
  run-handling defaults before the Colab steps go out. Nothing has trained. A
  later PR records the package and the outcome.

### 22.2 Pending: two report-only workflows

At 15:52 UTC the maintainer said "Yes" to (a) a report-only diagnosis of the
band runs' joint-space noise falls (K3) and (b) a scratch experiment on the tail
habit: P1 of §21.5 (a penalty on the policy mean beyond ±1), in two 1M-step CPU
continuations from stance5 s42's 4,194,304 checkpoint, P1 against a control, on
a scratch clone never pushed. Neither has a result yet, and neither enters a
verdict.

### 22.3 D-D40 and D-D36 amended

- **D-D40, amended 2026-10-22:** its build and package (§21.1), the runs and
  their runtime (§21.2), the outcome (§21.3), the saturation alarm (§21.4) and
  the s42 diagnosis that 5a asked for (§21.5).
- **D-D36, amended 2026-10-21:** walking pilot 2 rebuilt for stance3 Run 2's
  final and its branch pushed (§21.6); its parent kept for now by D-D41 (3).

### 22.4 What changes in §13–§21

- **§13 and §21.3, the band:** its next step is decided (D-D41): two more seeds,
  read under the three-of-five rule; not yet trained. No T. rex stance is
  certified, and the two-seed ask stands.
- **§20.4 and §21.6, walking pilot 2:** its parent stays stance3 Run 2's final
  for now, and s52's final is reconsidered after the replication. The pilot is
  not yet trained.
- **§21.4 and §21.5, the tail habit and the K3 falls:** two report-only
  workflows pending (§22.2).
- **§14:** no folder, file or branch is added to this repository by §22; the
  replication's package and the two workflows' outputs stay in the session's
  scratch.

## 23. Status 2026-10-24 (appended): the stance5 replication judged; the K3 diagnosis, the tail experiment and the K3 readouts; decision D-D42

*Appended 2026-10-24; §1–§22 above are unchanged, so §22's "not yet trained",
"neither has a result yet" and "the replication's package and the two
workflows' outputs stay in the session's scratch" are the wording of
2026-10-22.* The replication of D-D41 (§22.1) was made final and launched on
2026-10-22; its two runs trained to their final 12,001,280 steps on Colab L4 GPU
runtimes from 2026-10-22 to 2026-10-23 and were judged by their fixed criteria
with the replication's judge and scorer, the verdict of record written in the
run log's entry timed 13:04 UTC on 2026-10-24. This section records the package
and the runs (§23.1), the judging and the score (§23.2), the verdicts and the
decision-table line, which are results (§23.3), and the report-only gate,
certificates, extras and the saturation alarm on s80 (§23.4); the two
report-only workflows that §22.2 left pending (§23.5); the report-only K3
readouts the maintainer added on 2026-10-22 (§23.6); and the maintainer's
decisions of 2026-10-24 that followed, recorded as decision D-D42 with their
design pending (§23.7). A screen certifies nothing: the verdicts are the
criteria's, not a gate's, nothing here certifies a run, writes a result bundle
or promotes anything, and `main` and its digests are unchanged. The evidence is
in the folders `stance5/replication/`, `stance5/k3diag/` and `stance5/tailexp/`
of [stance_2026_10/](stance_2026_10/README.md), all added on 2026-10-24; the
replication's `results/` has its own
[README](stance_2026_10/stance5/replication/results/README.md). Paths in
backticks below are relative to `stance5/replication/` unless they start with
`../`, and every number below is read from `results/run_log.txt` (the session's
run log from the entry timed 17:46 UTC on 2026-10-22 on), the score files in
`results/scores/` or the files named.

### 23.1 The package and the runs

- **The package, made final** (17:46 UTC on 2026-10-22, after ten items a
  skeptic review raised were fixed): the criteria (`cbf98ba7…`, with a RUN
  HANDLING section for the maintainer's rules of 16:32 UTC), the judge
  `judge/judge_stance5r.py` (`16d39f7d…`, unchanged since §22.1), the sidecar
  `sigma_s70_s80.json` (`89778b3f…`) and the scorer `judge/score_stance5r.py`;
  `--validate` reproduces stance5's verdicts. At 17:47 UTC the session corrected
  the times of ten entries of 2026-10-22 in its log (the times §22 gives), and
  at 18:00 UTC the package was re-pinned for that correction, before any
  training step: the criteria `criteria.txt` (`a38536de…`), the scorer
  (`55c37ba7…`), the judge and the sidecar unchanged (the sidecar's own decision
  string keeps the pre-correction time, as the criteria note), 66 tests passing,
  and the corrected Colab steps sent (`colab_steps.txt`, `d646fc3c…`).
- **What the criteria fix** (`criteria.txt`): the recipe unchanged (arm LTSB at
  `cf80ecb1`, 12M steps on an L4, `n_envs` 4); Run 4 `stance5_LTSB_s70` (seed
  70) and Run 5 `stance5_LTSB_s80` (seed 80); the stance5 runs counted as
  recorded and re-checked by the scorer, which refuses to score if they do not
  reproduce; the control entering no decision; the judge, whose only changes
  from stance5's are the new seeds, the sigma file it records and its schema
  (`judge/judge_diff.txt`); the new seeds' joint-noise sigma, one vector, the
  element-wise mean of the frozen `control_sigma` of seeds 42, 52 and 60 (mean
  0.160713, against 0.160906, 0.160606 and 0.160626); K1–K4 as stance5's; and a
  decision table of three lines read once both runs have ended: 1, both hold
  (H = 3 of 5), adopt; 2, exactly one holds (H = 2), not adopted; 3, neither
  holds (H = 1), not adopted, the maintainer choosing the next round from
  stance5 line 4's menu on lines 2 and 3. The K3 pass fraction against the
  control is reported, not a veto, and drift or split is not read.
- **The runs.** The maintainer reported both started at 19:33 UTC on
  2026-10-22. The start-up check (19:41 UTC; two independent readers per run,
  byte-identical on all five files; `tools/startup_check5r.py`, report only, in
  the session's scratch) found each run's `provenance.json` on commit
  `cf80ecb1`, not dirty, with its run id and training seed, `n_envs` 4, r9,
  MuJoCo 3.10.0 as on stance5, SB3 2.9.0, gymnasium 1.3.0, numpy 2.1.3 and
  Python 3.13.15, and each `stage_config.json` equal to stance5 s52's but for
  the seed and the run label (the commit, `n_envs` 4, 12,000,000 steps, the
  hyperparameters `054ab483…`, the task `8c6998d2…`, the band and stagger keys,
  the collapse reference 4457.4 and an NVIDIA L4); `task_fingerprint.json` and
  `plant_identity.json` equal to s52's; and a zero-action reward of 4367.8, as on
  stance5.
- **The runtime, measured.** The `gpu` block of each `stage_config.json` reads
  an NVIDIA L4 (23.7 GB, CUDA 13.0). Each run's `progress.json` records one
  session, started (`session_started_at`) at 19:33:38 (s70) and 19:34:03 UTC
  (s80) on 2026-10-22 and ended at 12,001,280 steps (the file written at
  09:15:41 and 09:20:53 UTC on 2026-10-23), at 243.3 and 241.9 steps per second,
  with no warnings, so that each ran on the L4 throughout is inferred. A
  "STALE" ledger note on s80 at 01:21 UTC was a fetch-timing mismatch, not a
  resume: the next `progress.json` kept the same `session_started_at`
  (measured, 02:00 UTC).
- **The session.** The container restarted at about 01:05 and 01:51 UTC on
  2026-10-23, stopping the queue workers and, the second time, the running
  agents of two build workflows, which were relaunched or resumed; the scratch
  survived. At about 05:30 UTC on 2026-10-23
  every fetch agent hit the account's spend limit, and the session was idle
  until 12:06 UTC on 2026-10-24 (the container having rebooted again). To save
  usage after that, one combined fetch placed the J pairs at 8,388,608,
  10,485,760 and the final with the ledgers, `progress.json` and
  `gate_verdict.json`; the 1,048,576-step checkpoints 7,340,032, 9,437,184 and
  11,534,336 and the `best` and `robust_best` pairs were not fetched.

### 23.2 The judging and the score

- **Fetch and binding.** As in §21.2: each judged checkpoint was fetched
  read-only from Drive, each VecNormalize file transcribed twice and the copies
  compared (A = B at every fetch the log records), the zips checked with
  `testzip` (as the log records for the J and 1,048,576-step fetches), and the
  run records placed. The judge bound the plant, the seed, the task, the
  symmetry map and the run's `stage_config.json`, and read the new seeds' sigma
  from the sidecar, whose sha256 it pins.
- **The judged checkpoints (J)**: 4,194,304 (judged from 01:08 to 01:17 UTC on
  2026-10-23), 5,242,880 (02:16 to 02:26), 6,291,456 (from 03:16; logged at
  03:31), and 8,388,608, 10,485,760 and the finals (12:31 to 13:04 UTC on
  2026-10-24). The report-only 1,048,576-step checkpoints 1,048,576, 2,097,152
  and 3,145,728 were judged like J; 7,340,032, 9,437,184 and 11,534,336 were
  not (§23.1), nor the selection checkpoints S0, S1 and S3.
- **The score.** `results/scores/score_final.txt` and `.json` (sha256
  `9be2a231…` and `938da843…` as written; the JSON is kept gzipped), written by
  `score_stance5r.py --runs-ended`, the verdict of record, with no provisional,
  refusal or incomplete note. It checks the criteria's sha256 (matched), loads
  `score_stance5.py` and through it stance3's frozen scorer unchanged, reproduces
  the three stance5 runs' recorded verdicts, J cells, K1 failure modes, task and
  judge from their judge files, and reproduces the control's verdicts from
  stance4's score of record. Re-run on 2026-10-24 in the session's scratch as
  restored from the handoff, it gives the same verdicts, cells, H and line; it
  differs only where the handoff carried no file (the late-fall lines of the
  stance5 and control runs, and stance3 Run 2's reference cell in the parent
  comparison; `results/README.md`).

### 23.3 The verdicts: Run 4 holds; line 2

Per run, the judged checkpoints are 4,194,304, 5,242,880, 6,291,456, 8,388,608,
10,485,760 and the final; the offsets are the panel median of the both-feet
whole-foot centre of pressure, `x_c`, against the statue's 64.0 mm ahead of the
MTP III joints.

- **Run 4, `stance5_LTSB_s70` (seed 70): holds.** Clean 40, 40, 40, 39, 40 and
  40 of 40, `stance_quality/v2` PASS at all six. K2a and K2b 0 at every judged
  checkpoint but 8,388,608 (1 each); K2c passes (offsets −6.8, −9.0, −14.1,
  −15.4, −14.5 and −20.6 mm, a range of 13.7 mm). K3 passes (0 of 20
  joint-space noise falls at 6,291,456 and 1 at the final; own-sigma 0 and 0)
  and K4 passes (roll rail 0 and 0 of 40). It holds fore-aft.
- **Run 5, `stance5_LTSB_s80` (seed 80): fails K1 only.** Clean 40, 4, 14, 26,
  39 and 38 of 40; v2 FAIL at 6,291,456, PASS at the final. **K1 fails**: three
  judged checkpoints below 33 of 40, 5,242,880, 6,291,456 and 8,388,608, failing
  on the actuator-saturation rail in 36, 25 and 14 episodes (and one sole-tilt
  episode at 6,291,456); new failure modes against stance5's recorded ones, by
  the scorer. It failed for good at 5,242,880 (02:53 UTC on 2026-10-23), since a
  failed J checkpoint stays failed. K2 passes: K2a and K2b 0 but at the final (1
  each), a K2c range of 10.4 mm. K3 passes (1 and 2 of 20 joint-space falls,
  the bar 2; own-sigma 0 and 0) and K4 passes (1 and 1 of 40). It does not hold
  fore-aft, as K1 fails. By 10,485,760 it was back to 39 of 40 clean.

Per K test, the five band runs against the control (report only; the three
stance5 runs as recorded, §21.3, and the control as stance4's score of record,
§19.2):

| Test | s52 | s60 | s42 | s70 | s80 | Band runs passing | Control passing |
|---|---|---|---|---|---|---|---|
| K1, the gate held | PASS | PASS | FAIL | PASS | FAIL | 3 of 5 | 0 of 3 |
| K2, fore-aft held | PASS | PASS | PASS | PASS | PASS | 5 of 5 | 0 of 3 |
| K3, robustness | PASS | FAIL | FAIL | PASS | PASS | 3 of 5 | 2 of 3 |
| K4, roll | PASS | PASS | PASS | PASS | PASS | 5 of 5 | 2 of 3 |

**H = 2 of 5 (s52 and s70). The decision table's line is 2, final**: "Exactly
one of s70 and s80 HOLDS (H = 2 of 5): not adopted." The band is not adopted.
The scorer names s80's failing test (K1), its K1 failure modes and its
saturation alarm as the next round's inputs, and the maintainer chooses the next
round from stance5 line 4's menu: the tail saturation habit, the dose, the leg
exploration-noise floor. The K3 pass fraction, 3 of 5 against the control's 2 of
3, is below the control's, reported and not a veto (stance5's rule (b) is not
applied). None of the five band runs is classed drift or split (printed, not
read).

**The parent comparison** (report only; stance3's rule; the new runs' noise
falls use the sidecar's mean sigma): s52's final and stance3 Run 2's final
still tie (40 of 40, 0 noise falls), and the rule names both; behind them s70's
final (40 of 40, 1), the best stance4 final (s52's, 39 of 40, 0), s60's (39 of
40, 3), s80's (38 of 40, 2) and s42's (19 of 40, 14). Walking pilot 2's parent
stays stance3 Run 2's final (D-D41 (3)); the comparison does not choose it.

### 23.4 Report only: the gate, certificates, extras and the saturation alarm on s80

None of these enters a criterion (`results/scores/`, `results/notebook_gates/`,
`results/certificates/`, `results/saturation/`).

- **The 1,048,576-step checkpoints judged like J:** s70 39, 39 and 39 of 40 at
  1,048,576, 2,097,152 and 3,145,728 (one roll-rail episode each), s80 40, 40
  and 40; no noise falls in either.
- **The pre-registered readouts in the score:** K5 flags neither run; the
  late-fall fraction is above 5% only in the first 262,144-step bin (55.9% for
  s70, 48.5% for s80). The band readouts: no full-horizon episode of either run
  had a foot's window-mean whole-foot centre of pressure outside the band at a J
  checkpoint (one of s70's at 1,048,576 did), and the guard was never at 0 or on
  its ramp in any judged panel's full-horizon window foot-steps. The fresh, tail
  and common-noise panels on the finals were judged (their judge files are in
  the session's scratch) but are not read for this record, and the run log gives
  no numbers for them, for the 20-episode quick panels other than s80's at
  4,718,592 (6 of 20 clean, 14 on the saturation rail), for the per-term reward
  decomposition, the diagnostic panels or the stagger watches.
- **The notebook's own end-of-stage gate**, `stance_quality_heading/v1`, on
  each run's `robust_best`: both PASSED, s70's at 3,407,872 steps (40 of 40
  seeds, LCB 0.928) and s80's at 393,216 (38 of 40, 0.851). It judged an early
  checkpoint, not the finals the criteria judge.
- **The stage's six-heading certificate rolled offline on the finals**
  (13:14 to 13:54 UTC on 2026-10-24, in a clone of the screen branch at
  `cf80ecb1`, on the band-on task): s70 PASSED, 40 of 40 seeds clean at every
  heading (LCB 0.928); s80 PASSED, 38 of 40 (0.851; seeds 3047 and 3081 not clean
  at any heading). Neither is gate evidence: the band is not on `main`.
- **The saturation watch (D-D40 5a) on s80.** Its first flagged evaluation (5
  or more of 10 episodes on the saturation rail) was at 4,325,376 and the alarm
  (two flagged in a row) at 4,456,448, after 3,932,160 to 4,194,304 had been 10
  of 10 clean. At the alarm the ledger read 4, 3 and 5 of 10 clean at 4,325,376,
  4,456,448 and 4,587,520, flagged on 6, 7 and 5 episodes, with no falls and
  `actuator_force_frac_max` 0.29 to 0.31 (not force-limited, the log says).
  It was logged at 00:53 UTC on 2026-10-23 with the actuator not yet named, and
  the maintainer was told then (the log says so). Rollouts of the 4,718,592
  checkpoint named **`tail_3_pitch_act` at +1 (tail raised)** at 01:21 UTC: its
  command at 0.9 or more on 74% of window steps (mean +0.91), and 14 of 20
  episodes over the gate's bar; at 5,242,880, 92% (+0.97) and 19 of 20; at
  6,291,456, 63% (+0.81) and 12 of 20; no rolled episode fell. In all, 37 of
  s80's 91 evaluations were flagged: 35 in five alarm runs (4,325,376–6,815,744,
  20; 7,340,032–7,471,104, 2; 7,733,248–7,864,320, 2; 8,388,608–9,306,112, 8;
  9,699,328–9,961,472, 3) and 2 alone (7,077,888 and 10,223,616). s70 had no
  flagged evaluation (0 of 91). No run was stopped or changed on the alarm.
- **The tail habit across the runs.** That is the actuator and sign of stance5
  s42 (§21.4), stance4 s42 (§21.5) and stance3 Run 2; with s80, four of the ten
  runs so far on L+T-based recipes have raised a saturation alarm, all on this
  actuator and sign (counted from §21.5 and this section).

### 23.5 Report only: the K3 diagnosis and the scratch tail experiment (§22.2)

Both workflows finished at 21:04 UTC on 2026-10-22; their reports were written
to file by the session, each with its skeptic checks first (where a check
differs from the synthesis, the check governs), and are in `../k3diag/` and
`../tailexp/`. Neither changes a verdict.

- **The K3 diagnosis** (`../k3diag/report.md`; four checks, three of which
  corrected the synthesis). Measured and checked: stance5 s42's joint-space
  noise falls are not the tail habit (holding `tail_3` at 0 leaves 0, 9 and 12
  falls against 0, 9 and 13 at 6,291,456, 10,485,760 and the final; tail-only
  noise gives 0 falls at every checkpoint). Hip-pitch noise is the main cause
  (alone 8, 3, 9 and 15 of 20 at 7,340,032, 8,388,608, 10,485,760 and the final;
  the full noise without hip pitch 0, 0, 0 and 2), mostly its left-right
  differential part at the judge's real amplitude (5, 2, 7 and 15) against its
  common part (0, 0, 0 and 1); control s42's final needs knee and hip-pitch
  noise together. The falls are slow topples (per-slot medians of 0.9 to 1.5 s
  from onset to the end) after a heading-frame stagger of at least 81 mm, which
  comes mostly from the pelvis yawing over nearly planted feet; associated, not
  a threshold. The policy's direct response to a left-right hip-pitch
  difference halves between 6,291,456 and 7,340,032 (−1.04 to −0.51 per rad),
  but restoring it from outside leaves the falls (7, 9 and 12 of 20 at 7.3M,
  10.5M and the final): a correlate, not the mechanism, which is open. Not
  independently checked: the band-against-control paired falls (43 against 11)
  and the kick probe. The report's reading (inferred): a seed-42 effect shared by
  both arms is supported, a band trade-off is possible but not shown, and s60's
  3 of 20 is within luck.
- **The scratch tail experiment** (`../tailexp/report.md`; three checks, one
  of which corrected the headline). P1 of §21.5, a quadratic penalty on the
  policy's Gaussian mean beyond ±1, λ 41, in a 1M-step CPU continuation from
  stance5 s42's 4,194,304 checkpoint, against a paired control (coefficient 0)
  from the same checkpoint: one seed, one pair. Measured and checked: P1's
  tail_3 share at |cmd| ≥ 0.99 was lower than the control's at 8 of 8 periodic
  checkpoints and at the final (means 0.26 against 0.86), not monotone, and it
  relapsed to 0.470 and 0.488 at 5,242,880 and the final; P1 did not keep the
  deterministic mean under +1 (beyond +1 on 21–46% of window steps at 7 of 9
  checkpoints), only the training batches' (2.5% of rows against 27.5%). The
  original no-penalty GPU run read 0.171 at 5,242,880, below P1's 0.470, so the
  effect against training alone is not established. No cost was measured, but
  the window ends at 5,242,880, before the K3 slot and before s42's noise falls
  began. The code was a scratch clone, `scratch_tailfix` (branch
  `scratch/tail-mean-bound` at `1d2d22b4`, on `cf80ecb1`, push URL disabled):
  it was never pushed and is not in this repository; the report describes it. A
  real round would need an opt-in configuration key, criteria fixed before
  training and the maintainer's approval, since the bound changes the PPO
  training loss.

### 23.6 Report only: the K3 readouts addendum and its results

At about 21:05 UTC on 2026-10-22 the session told the maintainer the K3 results
and asked whether to add report-only differential hip-pitch readouts to the
replication; the maintainer said yes, and the addendum
(`readouts_addendum_k3.txt`, sha256 `7a79e9f6…`) was written at 22:55 UTC,
before any J checkpoint of the replication existed. It changes neither the
criteria nor the judge, scorer or sidecar, says it is post hoc, and enters no
verdict, decision line or tie-break. The tool `k3readouts/k3readout5r.py`
(`d0aed5a1…`) and its runner (`5e3c66cd…`) were validated against stance5's
skeptic-checked values (223 checked rows, 0 mismatches; `k3readouts/VALIDATION.md`)
after two reviews found 13 minor points and none blocking. Measured (the four
valid readouts, `k3readouts/out/`; n = 20 per condition):

| | s70 6,291,456 | s70 final | s80 6,291,456 | s80 final |
|---|---|---|---|---|
| Full-noise falls of 20 (the judge file's) | 0 | 1 | 1 | 2 |
| Hip-pitch noise only / without hip pitch | 0 / 0 | 0 / 0 | 1 / 0 | 1 / 0 |
| Differential / common part of the hip-pitch noise | 0 / 0 | 0 / 0 | 0 / 0 | 1 / 0 |
| ±0.3 / ±0.2 differential hip kick, falls of 20 | 18 / 0 | 20 / 0 | 20 / 0 | 20 / 1 |
| Direct response to a left-right hip-pitch difference (per rad) | −1.023 | −0.428 | −0.484 | −0.012 |
| Own hip-pitch sigma | 0.0220 | 0.0146 | 0.0212 | 0.0149 |

The references, rolled by the same tool: stance5 s42's final fell 13 of 20
(hip-pitch only 15, the differential part 15), s52's 0 and s60's 3; every
stance5 band final fell 20 of 20 to the ±0.3 kick, and the stance4 control
finals 0, 1 and 11 of 20 (the K3 diagnosis). Inferred, at this n: neither new
run showed s42's K3 collapse, which supports a seed-42 effect for those falls;
all five band finals fall to the sharp kick against 0 to 11 of 20 for the
control finals, a consistent band-arm property that no criterion reads; and the
direct response does not predict the falls (s80's final is near 0 and falls 2
of 20), in line with the diagnosis' finding that it is a correlate.

### 23.7 The maintainer's decisions of 2026-10-24 (D-D42), design pending

The round was closed on 2026-10-24 in the session that ran it, and its files
were carried to a new session on a handoff branch that is never merged (the
maintainer's decisions in the run log's entries timed 23:22 UTC on 2026-10-22
and 12:55 UTC on 2026-10-24); checkpoints stay on Drive. In the new session, at 15:20 UTC on
2026-10-24, the maintainer decided, as the run log has them in short:

- **(1) The next round targets the tail saturation habit**, stance5 line 4's
  first item. The round is designed with fixed criteria and brought back to the
  maintainer for approval before any training. At 15:27 UTC an options study
  started, with no training and no Drive fetch; its brief comes to the
  maintainer before any design is fixed. Nothing is designed or fixed yet.
- **(2) The training-history report goes to a pull request at scope (a)**, the
  standalone report as built: report-only pages drawn from the run ledgers
  (no rollouts), built on 2026-10-22 and 2026-10-23 as the first piece of the stage
  report's design (the maintainer's decisions of 2026-10-18), on a local branch
  `claude/run-training-report` (commits `4a783bae` and `50fd8f60`, not pushed
  then; 158 tests passing and ruff clean on `environments/` by the log's 02:54
  UTC entry), carried as patches and applied in the new session (`1e0da30`,
  `356bf3f` on `1b71f00`), its tests pending when the copied log ends (15:27
  UTC on 2026-10-24). It is not part of this change.
- **(3) An actuator saturation report**, both during training and for finished
  models, from the ledger: report-only telemetry naming the saturating actuator
  and its sign on each evaluation-episode row, drawn over training by the
  training-history report and read for finished models from their evaluation
  rows. The evaluation ledger has no per-actuator column today (§21.5), so the
  actuator was named by the scratch rollouts of §23.4. It is to be built; not
  part of this change.
- **(4) The records PR, documentation only**: this change.

[BEHAVIOR_RECIPES_PLAN.md](../BEHAVIOR_RECIPES_PLAN.md) §6.2 records them as
decision D-D42, taken with its design pending, and D-D41's outcome (§23.3) as
D-D41's dated amendment. Walking pilot 2 is not launched (the run log records
no launch), and its parent stays stance3 Run 2's final.

### 23.8 What changes in §13–§22

- **§13, §21.3 and §22.1, the band:** replicated; one of the two new runs holds,
  H = 2 of 5, and the decision table's line is 2, final: the band is not
  adopted. No T. rex stance is certified, and the two-seed ask stands.
- **§21.4 and §21.5, the tail habit:** it appeared again, on s80 (§23.4); the
  next round targets it, its design and criteria to be fixed and approved
  before any training (§23.7).
- **§22.2, the two report-only workflows:** done, report only (§23.5); and the
  K3 readouts added on 2026-10-22, report only (§23.6).
- **§20.4, §21.6 and §22.1 (3), walking pilot 2:** its parent stays stance3 Run
  2's final; s52's final still ties it under the parent rule. The pilot is not
  yet trained.
- **§14, this repository:** the folders `stance5/replication/` (with
  `results/` and `k3readouts/`), `stance5/k3diag/` and `stance5/tailexp/`, with
  their sources and sha256 in `SOURCES.txt` and `SHA256SUMS` (now 418 files).
- **§14, branches:** `claude/screen-trex-r9-armlts-copband` at `cf80ecb1` and
  `claude/pilot-walk-r9-b` at `8c5478a6` stay on GitHub, never merged; the
  handoff branch `claude/handoff-stance5r` is on GitHub, never to be merged. The
  scratch clone `scratch_tailfix` is not pushed, and the training-history
  report's branch `claude/run-training-report` was local when the copied log
  ends (15:27 UTC on 2026-10-24).
- **§14, Google Drive:** the two runs' folders `stance5_LTSB_s70/` and
  `stance5_LTSB_s80/` under
  `MyDrive/mesozoic-labs/logs/tyrannosaurus_rex/anatomical/ppo/` hold their
  checkpoints; each judged one's sha256 is in its judge file, in the session's
  scratch.
- **§14, session scratch only:** the judge files
  (`scratchpad/foot/stance5/replication/judged/`), the certificates'
  per-episode files (`scratchpad/foot/stance5/replication/cert/`), the runs'
  fetched records and ledgers (`scratchpad/foot/stance5/replication/runs/`), the
  start-up checks (`scratchpad/foot/stance5/replication/startup/`), the tools
  and their logs (`scratchpad/foot/stance5/replication/tools/` and
  `k3readouts/logs/`), and the K3 diagnosis' and tail experiment's rollouts,
  tables, runs and verifiers (`scratchpad/foot/stance5/k3diag/` and
  `scratchpad/foot/stance5/tailexp/`).
