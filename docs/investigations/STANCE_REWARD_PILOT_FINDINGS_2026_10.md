# Stance reward pilots: summarized findings and recommendations

**Date:** 2026-10-10

**Status:** completed research screen; follow-up proposals pending validation

The T-Rex centre-of-pressure (CoP) margin penalty is the strongest candidate
to carry forward: clean evaluations increased from **55/120 to 96/120**,
with improvements in all three paired training seeds. **None of the 15
pilots passes the original `stance_quality/v2` gate.** Neither velociraptor
candidate fixes its required distributed foot support.

This investigation summarizes completed outcomes and proposes controlled
follow-ups. The scope is documentation, with summary statistics sufficient
to distinguish a promising direction from an adoption result.

## 1. Study scope and aggregate findings

The five arms each used three fresh training seeds and 1,048,576 steps per
run, followed by deterministic evaluation of the final checkpoint on 40
reset seeds. The original 11M-step T-Rex and 6M-step velociraptor schedule
budgets were retained. These are short screens, not full-budget trials.

The research source is pinned at
[`b2c04be76d7f94518a3f6b2ed88e7120c98aa9c2`](https://github.com/kuds/mesozoic-labs/tree/b2c04be76d7f94518a3f6b2ed88e7120c98aa9c2/docs/investigations/stance_rewards_2026_10).
It uses T-Rex physics r8 / policy interface r13 and velociraptor physics
r3 / policy interface r11. The recorded results were reconciled with the
completed manifest and comparison. Original gate judgments were
independently reproduced over 24 panels / 960 episode metric rows,
including calibration and matched home-controller panels.

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

The T-Rex penalty uses weight **0.10**, safe fraction **0.65** of the pad
half-length, and the worst foot's bounded squared deficit after settling.
Fore-aft CoP violations decrease from **62/120 to 21/120**. Sole-tilt and
corner-lift violations do not improve, and survival is essentially
unchanged. The best candidate panel's clean LCB is **0.725**, below the
required **0.80**. At least 37/40 clean evaluations are needed for that
bound, alongside the other gate requirements. Original-task mean return
also declines slightly; physical conjunction remains the selection target.

For velociraptor, raising support-conditioned alive fraction from **0.20
to 0.50** provides inconsistent survival changes. The contact-quality arm
combines airborne, impact and contact-coverage penalties. It lowers
recorded impacts and airborne counts, but its shorter episodes limit
exposure to later failures, and the bundled terms cannot be attributed
separately. All **360 velociraptor evaluations** fail both required
contact-site duty and coverage.

Both plants' zero-action home controllers are **40/40 clean** on
calibration and matched evaluation panels. The plants can meet the stance
gate near their home equilibrium. This motivates testing learning and
load feedback first; robustness to disturbances remains a separate question.

## 2. Recommended follow-ups

Keep the current gate and network architecture `[512, 256]` as controls.
The following changes are hypotheses to test, with one cause isolated per
comparison.

| Priority | Proposed change | Evidence needed before adoption |
|---|---|---|
| 1 — T-Rex | Carry CoP weight 0.10 and safe fraction 0.65 into a longer paired trial. | Repeatable per-seed clean/LCB and CoP gains without increased falls, sole tilt or corner lift. |
| 1 — Velociraptor | Keep alive support fraction 0.20; compare `log_std_init = -1.5` with default 0. | An exploration-only comparison under the same reward and schedules. This smaller initial exploration has not been tested by these pilots. |
| 2 — Velociraptor | Test bounded continuous support shaping for each required contact site. | Better weakest-site floor-normal load, duty and coverage, with acceptable survival, impacts and command/force saturation. Calibrate targets to each site's home load rather than equalizing unequal anatomical loads. |
| 3 — Velociraptor | Expose individual contact-site loads to the policy. | Separate observation-only and reward-only comparisons before combining them; validate channels against floor-contact truth and retrain on the revised policy interface. |
| 4 — Velociraptor | Split airborne, impact and coverage penalties into individual ablations. | Distributed support and survival gains, with duration reported alongside impact/airborne metrics. |
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

Predeclare paired training seeds, schedule budgets and an untouched
evaluation panel. The panel inspected for this study should not become an
untouched selection panel for tuning. Report each training seed's outcome
as well as descriptive aggregates.

Keep original-task reward, candidate reward and physical gate judgments
separate. Report clean count/LCB, survival, failure reasons, weakest
foot/site loads, CoP/tilt/corner lift where applicable, impacts, airborne
substeps, raw command saturation and actual force clipping. Include
episode-duration distributions to account for early termination.

Before promotion, run the unchanged current gate and its consecutive-panel
and multi-seed certification procedure, including the production fresh-panel
and heading-probe requirements in NEXT_STEPS. Recheck clean home controls
and known stance hacks. Bound load rewards so floor strikes or contact
overloads cannot buy support credit. Negative mean returns from an
exploration proxy include terminal costs and do not establish that early
failure is preferred; nonterminal rewards must be considered separately.

Added observation channels require a policy-interface revision and fresh
policy/normalizer training. Reward changes require a task fingerprint and
review of statue/recovery inheritance constants. Physical edits require
home-equilibrium, contact and preload recalibration, plus the compatibility
procedure in [PLANT_CONTRACT](../PLANT_CONTRACT.md).

This screen establishes neither long-budget convergence nor cross-species
transfer. It contains no push test; disturbance response belongs to Recovery.
The recommendations apply to the two tested species and need revalidation
after any observation or plant change.
