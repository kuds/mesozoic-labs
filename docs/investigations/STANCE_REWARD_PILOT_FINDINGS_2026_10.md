# Stance reward pilots: summarized findings and recommendations

**Date:** 2026-10-10

**Status:** completed research screen; follow-up proposals pending validation

The T-Rex centre-of-pressure (CoP) margin penalty is the strongest candidate
to carry forward: clean evaluations increased from **55/120 to 96/120**,
with gains in all three paired training seeds (+1, +26, +14; n = 3).
**None of the 15 pilots passes the original `stance_quality/v2` gate.**
The base T-Rex reward is CoP-indifferent: a pad-edge policy and a
gate-passing one earn the same, both above the statue (§1.2). The
velociraptor pilots mostly hop and fall under default-scale exploration,
and digit IV stays unloaded even with both feet down (§1.4); neither
velociraptor candidate fixes the falls or the digit-IV deficit (the
contact-quality bundle removes the hop but shortens episodes).

This investigation summarizes completed outcomes and proposes controlled
follow-ups. The scope is documentation, with summary statistics sufficient
to distinguish a promising direction from an adoption result.

## 1. Study scope and aggregate findings

The five arms each used three fresh training seeds and 1,048,576 steps per
run, followed by deterministic evaluation of the final checkpoint on 40
reset seeds. The original 11M-step T-Rex and 6M-step velociraptor schedule
budgets were retained. These are short screens, not full-budget trials;
§1.3 lists full-budget production runs on the same tasks as a reference.

The research source is pinned at
[`b2c04be76d7f94518a3f6b2ed88e7120c98aa9c2`](https://github.com/kuds/mesozoic-labs/tree/b2c04be76d7f94518a3f6b2ed88e7120c98aa9c2/docs/investigations/stance_rewards_2026_10),
the head of the unmerged branch `codex/stance-reward-experiments`; its
`environments/` and `configs/` match main at `10f859f`, and its status
files describe an earlier local-CPU launch, not this batch. It uses T-Rex
physics r8 / policy interface r13 and velociraptor physics r3 / policy
interface r11. The Colab runner, `colab_runner.py`, is not in git; each
run's results record its `adapter_sha256` (`2191717f…`). The Drive batch
folder is `experiments/stance_reward_pilots_20261009_colab_v1`, one run
folder per pilot named `<species>_<arm>_seed<N>`. The runner kept no
learning curves, no termination reasons and only the last two checkpoints
of each run; the follow-up must record all three. The recorded results
were reconciled with the completed manifest and comparison. Original gate
judgments were independently reproduced over 24 panels / 960 episode
metric rows, including calibration and matched home-controller panels.

| Species / arm | Clean evaluations, out of 120 | Full horizon, out of 120 | Passing pilot panels, out of 3 |
|---|---|---|---|
| T-Rex control | 55 | 118 | 0 |
| T-Rex CoP margin | 96 | 117 | 0 |
| Velociraptor control | 0 | 20 | 0 |
| Velociraptor support fraction | 0 | 18 | 0 |
| Velociraptor contact quality | 0 | 10 | 0 |

These are descriptive totals over three trained policies per arm. The
120 resets are not 120 independent training replicates, and pooling them
does not satisfy the curriculum's per-panel or multi-seed certification.

Both plants' zero-action home controllers are **40/40 clean** on
calibration and matched evaluation panels. The plants can meet the stance
gate near their home equilibrium. This motivates testing learning and
load feedback first; robustness to disturbances remains a separate question.

### 1.1 T-Rex per-seed results

The T-Rex penalty uses weight **0.10**, safe fraction **0.65** of the pad
half-length, and the worst foot's bounded squared deficit after settling.
Fore-aft CoP violations decrease from **62/120 to 21/120**. Sole-tilt and
corner-lift violations do not improve, and survival is essentially
unchanged. The best candidate panel's clean LCB is **0.725**, below the
required **0.80**. At least 37/40 clean evaluations are needed for that
bound, alongside the other gate requirements. The arms' original-task mean
returns (3708.4 control, 3679.3 CoP margin) differ by about one additional
fall: three early ends against two, each costing a panel about 80.
Physical conjunction remains the selection target. Per training seed, on
the evaluation seeds 23042–23081:

| Training seed | Clean, control → CoP margin | Clean LCB | Fore-aft CoP failures |
|---|---|---|---|
| 42 | 30 → 31 | 0.613 → 0.640 | 10 → 6 |
| 48 | 5 → 31 | 0.051 → 0.640 | 35 → 9 |
| 50 | 20 → 34 | 0.361 → 0.725 | 17 → 6 |

At n = 3 the gains are directional, not established: one-sided paired t
p = 0.099, exact rank-sum p = 0.05, sign test p = 0.125 (its floor at three
pairs); the pooled count treats episodes clustered in three policies as
independent. Seed 48's control 5/40 is not a collapse but a stable stance
with one pad on its front edge (CoP median 0.990, about 9 mm of drift, no
saturation); the D-D28 bar alone refuses 31 of its 35 unclean episodes.

### 1.2 T-Rex reward ordering

Calibration seeds 3042–3081, same task (`task_sha256 4c341ba1…`). Rewards,
clean counts and failure counts are from the pinned
`calibration_summary.json`, CoP readings from the batch's
`calibration_trex.json`, and seed 52 entirely from its own gate report;
the CoP-priced column re-prices the same trajectories with `cop_margin`:

| Controller (selected checkpoint) | Base reward | CoP-priced | Clean | Gate notes |
|---|---|---|---|---|
| Zero-action statue | 3756.79 | 3756.70 | 40/40 | CoP at most 0.362 |
| Seed 42, `20261008_163256` (~10.15M) | 3783.39 | 3712.94 | 0/40 | all 40 fail CoP (median 0.992) |
| Seed 48, `20261008_163410` (0.55M) | 3774.27 | 3770.30 | 36/40 (LCB 0.786) | 3 CoP and 1 tilt failures |
| Seed 52, `20261009_155723` (5.55M) | 3782.2 | not computed | 39/40 (LCB 0.887), PASS | 1 settle-width failure; CoP median 0.703, worst 0.797 |

**The base T-Rex reward is CoP-indifferent.** The edge policy and the
passing policy earn the same (1.2 apart), both about 26 more than the
statue, so the statue is not the reward optimum. That is the case for a
CoP term: `cop_margin` prices the edge policy below the statue. Measured
mechanisms to test alongside it:

- `leg_home_pose` is centred on the authored keyframe, not the settled
  statue, which earns 471 of the term's 500 per episode; re-scored with a
  settled reference, the statue beats every tested scripted policy,
  including seed 52's mean action held open-loop (the trained checkpoints
  were not re-scored). Most of that pull is a benign hip-pitch correction,
  so this complements a CoP term.
- The one-sided nosedive term's natural pitch (0.027 rad) is nose-up of
  the r8 statue, costing it about 7 per episode that nose-up trim recovers.
- `foot_flatness` sees tilt only (tolerance 3° against the gate's 2° bar),
  so a flat pad loaded on its front edge costs nothing through it.

Seed 52's pass does not replicate: re-rolled from the bundle's checkpoint
during the 2026-10-10 review (also the source of the settled-reference
re-scoring), it is 32/40 clean (LCB 0.668) on the D-D27 re-judge seeds
7042–7081, with 7 CoP failures and 1 nosedive; the statue is 40/40.

### 1.3 Full-budget reference runs

These production runs trained the base reward at full budget on the
pilots' tasks (T-Rex `4c341ba1…`, velociraptor `d19abe5c…`): the three
T-Rex runs of §1.2 and the two velociraptor runs below. They are a
**reference**, not the control distribution of the follow-up, which trains
fresh arms (§2). Each panel judges the reward-selected checkpoint (best
eval mean − std), so T-Rex seed 48's 36/40 is a 0.55M checkpoint. Seed
42's selected checkpoint drifted with long training (30/40 in the same
seed's 1M pilot, 0/40 at ~10.15M), and so did seed 52's final 11M
checkpoint: D-D28 fails it on 33/40 of 3042–3081 at the selected
checkpoint's reward (3782.05 against 3783.09 on the same 39 standing
episodes, both re-rolled in the review), so seed 52's pass rests on the
5.55M selection.

| Velociraptor run | Seed | Selected checkpoint | Clean on 3042–3081 | Failure pattern |
|---|---|---|---|---|
| `20261007_132051` | 42 (commit `46e0b7c`) | ~5.15M of 6M (best mean eval at 6.0M) | 0/40, full horizon 40/40 | ~10 Hz hop on 25/40; 13 stand still yet fail (a parked actuator, digit IV unloaded) |
| `20261009_155731` | 48 | ~5.9M of 6M | 0/40, full horizon 40/40 | ~10 Hz hop on digit III, all 40 |

### 1.4 Velociraptor: hop and fall, and an unloaded digit IV

The dominant measured failure is hopping and falling: 312 of the 360
evaluations end early. Control panels have no foot down on a median
0.54–0.69 of window steps, with median window peaks of 11–12.6 body
weights; a deterministic replay of control seed 50 during the review
ended 29/40 episodes in tail contact. Both 6M production runs are 0/40
with a ~10 Hz hop (§1.3). The per-digit deficit is real too: in control
seed-42 episodes with both feet down on 0.76–0.93 of the window, digit-IV
duty is 0.0–0.008 (digit III and metatarsus 0.83–0.98). All **360
velociraptor evaluations** fail required contact-site duty and coverage.

Raising the support-conditioned alive fraction from **0.20 to 0.50**
provides inconsistent survival changes. The contact-quality arm combines
airborne, impact and contact-coverage penalties. It removes the whole-body
hop (median flight 0.03–0.11), but its episodes are shorter (median
140–217 steps), support-site duty stays near zero, and the bundled terms
cannot be attributed separately.

Every velociraptor arm ran at SB3's default action std 1.0 (no
`log_std_init`); the learned std is still about 0.97 at 1M steps. The
untrained network, run deterministically, stands like the statue (20/20
clean), but under i.i.d. action noise the statue keeps full horizon on
20/20 episodes at σ 0.07 and on 0–2 of 20 at σ 0.10. The asymmetric
home-residual action map turns zero-mean noise into a backward lean
(about −17.9σ° at the hip, +13σ° at the ankle) that alone, held at its
σ 0.22 value, tips the statue onto its tail. These are the 2026-10-10
review's measurements. The reward ranks the statue first (2842.8 against
2395.9–2443.9 for the 6M runs): the blocker is reaching the standing
basin, not the reward's optimum.

## 2. Recommended follow-ups

Keep the current gate and network architecture `[512, 256]` as controls.
The following changes are hypotheses to test, with one cause isolated per
comparison.

| Priority | Proposed change | Evidence needed before adoption |
|---|---|---|
| 1 — T-Rex | Carry CoP weight 0.10 and safe fraction 0.65 into a full-budget paired trial (§3). | Repeatable per-seed clean/LCB and worst-pad CoP gains on both panels, without increased falls, sole tilt or corner lift. |
| 2 — T-Rex | Test a settled `leg_home_pose` reference and a recalibrated nosedive natural pitch, alone and with the CoP term. | Re-priced calibration in which the statue out-earns every known edge policy (production seed 42, seed 52's final checkpoint, pilot control seed 48); statue-derived constants re-measured. |
| 1 — Velociraptor | Paired exploration sweep under the unchanged reward: default `log_std_init` 0 against −1.5, about −2.3 and about −3.0 (σ 0.22 / 0.10 / 0.05). A symmetric or narrower residual map is the policy-interface option. | Whether the deterministic policy is still standing at each checkpoint, with learned std, full horizon, flight fraction and clean count. The review's short probes were ambiguous between −1.5 and −3.0: a sweep, not a known answer. |
| 2 — Velociraptor | Price airborne substeps and floor impact as individual terms. | The hop removed without shorter episodes, duration reported alongside impact/airborne metrics. The contact-quality bundle that included them removed the whole-body hop but not the support-site deficit. |
| 3 — Velociraptor | Test bounded continuous support shaping for each required contact site. | Better weakest-site floor-normal load, duty and coverage, with acceptable survival, impacts and command/force saturation. Calibrate targets to each site's home load rather than equalizing unequal anatomical loads. |
| 4 — Velociraptor | Expose individual contact-site loads to the policy. | Separate observation-only and reward-only comparisons before combining them; validate channels against floor-contact truth and retrain on the revised policy interface. |
| Conditional plant work | Audit toe-IV preload, contact compliance and ankle/toe damping. | A measured failure persisting with a quiet policy; separate actual actuator-force clipping from raw normalized command saturation before geometry or strength changes. |

The T-Rex follow-up respects the heading migration prerequisite in
[NEXT_STEPS](../NEXT_STEPS.md): fresh production training waits for the
agreed heading-free interface. These pilot findings apply to the pinned
legacy observation. After migration, train both control and CoP arms
fresh on the same new interface and revalidate the effect. Any optional
source-frozen extension remains research.

Relevant code supports the velociraptor experiment ordering:

- The [raptor observation](../../environments/velociraptor/envs/raptor_env.py)
  sums digit-III, metatarsal and digit-IV touch signals into two foot
  totals. Individual load channels can expose a missing support site.
- The research coverage penalty averages loaded-site indicators within a
  foot. That can dilute one unloaded digit; a continuous weakest-site
  deficit is a candidate for better feedback, subject to calibration.
- The recorder's [`max_actuator_saturation_fraction`](../../environments/shared/gait/stance_metrics.py)
  measures raw normalized actions, not actual actuator-force clipping.
- The [raptor home keyframe](../../environments/velociraptor/assets/raptor.xml)
  already includes digit-IV preload and unequal anatomical support loads.
  Quiet home success does not justify stronger actuators or altered
  geometry without a diagnosis.

## 3. Validation criteria and limits

Predeclare each follow-up. Every arm trains at the full schedule budget
(11M T-Rex, 6M velociraptor) on at least two paired training seeds (the
T-Rex stance's declared `certification_seeds = 2`, applied here to both
species), more where budget allows. Each selected checkpoint is judged
with the unchanged gate on 3042–3081 and re-judged on 7042–7081. Primary
endpoint: per-seed clean count and LCB. Secondary: falls; worst pad CoP,
sole tilt and corner lift (T-Rex); full horizon, flight fraction and the
weakest support site (velociraptor). The T-Rex arms train after the
heading-free interface and must also pass its heading certificate. The
23042–23081 panel inspected here should not become a selection panel for
tuning.

Keep original-task reward, candidate reward and physical gate judgments
separate. Report clean count/LCB, survival, failure reasons, weakest
foot/site loads, CoP/tilt/corner lift where applicable, impacts, airborne
substeps, raw command saturation and actual force clipping. Include
episode-duration distributions, learning curves, learned std, termination
reasons and sparse checkpoints. Before promotion, also run the
consecutive-panel procedure and the production fresh-panel requirements
in NEXT_STEPS. Recheck clean home controls and known stance hacks. Bound
load rewards so floor strikes or contact overloads cannot buy support
credit. Negative mean returns from an exploration proxy include terminal
costs and do not establish that early failure is preferred; nonterminal
rewards must be considered separately.

Added observation channels require a policy-interface revision and fresh
policy/normalizer training. Reward changes require a task fingerprint and
review of statue/recovery inheritance constants. Physical edits require
home-equilibrium, contact and preload recalibration, plus the compatibility
procedure in [PLANT_CONTRACT](../PLANT_CONTRACT.md).

This screen establishes neither long-budget convergence nor cross-species
transfer. It contains no push test; disturbance response belongs to Recovery.
The recommendations apply to the two tested species and need revalidation
after any observation or plant change.
