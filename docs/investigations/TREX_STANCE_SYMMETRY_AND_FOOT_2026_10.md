# T. rex stance after run A: symmetry, the r10 foot and the forward lean

**Date:** 2026-10-17 (the experiments of 2026-10-12 to 2026-10-17)

**Status:** completed screens and analyses, the technical record of the T. rex
stance work since the blog post of 2026-10-11. The walking pilot of §9
(decision D-D36, [BEHAVIOR_RECIPES_PLAN.md](../BEHAVIOR_RECIPES_PLAN.md) §6.2)
is pre-registered, **launched on Colab on 2026-10-17 and not yet judged**; its
outcome will be appended to this note as a dated note when it is judged. The
evidence, frozen copies of the scripts that decided each verdict, and their
sums are in [stance_2026_10/](stance_2026_10/README.md).

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
