# Standalone robot design and regression review

**Date: 2026-10-25.** Scope: the complete mini/medium T-Rex change set in
PR #635, reviewed from `42a7e989` against main `9f2355a3`. The branch incorporates
that main revision, including its actuator-force telemetry work. This is a
review record, not a landing record or a hardware acceptance certificate.

## Findings and repairs

| Finding | Change | Verification |
|---|---|---|
| The mini passes six seconds but creeps during a longer hold | Leg position gains ×0.5 and velocity gains ×0.25; no geometry, mass, torque-cap or support change | Both configurations hold 30 s, including small roll/pitch kicks; a half-timestep regression also runs |
| Unsupported probes could miss gravity compensation or fixture constraints; some probes cleared display support before inspecting it | Shared `robot_validation` guards inspect the reset trial before any display-only support change | Negative controls for fixed/mocap body and site welds, connect constraints, disabled physics, gravity compensation and applied forces |
| Floor-name/touch-only checks could miss another object supporting the robot | Explicit robot/foot/floor contact classification at physics substeps | Fixture, non-foot and stacked-foot negative controls; actual settled floor loads in reports |
| Pre-commit covered environments only while CI also checked the examples | Local Ruff/mypy and both CI typing jobs cover matching source directories, including sizing scripts | Workflow-contract tests verify scope and asset-only triggers |
| Moving the old walking report broke both sizing calculators and downstream cost reconciliation | Pin the dated similarity screens to `baseline_rev10/walking_validation.json`; label that historical basis | Run every calculator in an isolated copy without the current gait report; compare regenerated numbers and hashes |
| Current numerical evidence could remain stale after a source edit | Refresh current reports and bind them to their source/model/helper hashes | Source-fingerprint regressions, separate from the actual physics tests |
| Medium animation exceeded the repository's 500 KiB limit | Re-render identical physics poses with a smaller nondithered palette | File-size check; retained display labels and timing |

The concepts remain checkout-only examples. Their model names, tests and docs use
the existing species conventions; shared support checks live in the shared
package. No registered plant, policy interface, task recipe, manifest entry or
frozen MJX module is changed by these repairs. Variant registration remains the
separate [architecture proposal](../hardware/ROBOT_VARIANTS_AND_BUILD_PLAN.md).
The original mini and 51.5 cm medium baselines remain historical references.

## Balance evidence

All rows below use a free root, Earth gravity, real sole-to-floor support and no
applied external support. The kick cases add a specified 0.15 rad/s root angular
velocity at one second, not an ongoing force or controller assistance.

| Current configuration | Trial length | Maximum root drift | Maximum torso tilt | Highest settled leg RMS |
|---|---:|---:|---:|---:|
| Mini, fixed tail | 30 s | 1.54 mm | 1.72° | 0.0529 Nm |
| Mini, added tail centered | 30 s | 1.76 mm | 1.99° | 0.0588 Nm |
| Medium, quiet stance | 30 s | 4.86 mm | 2.61° | 1.79 Nm |
| Medium, standing gestures | 30 s | 4.85 mm | 2.61° | 1.80 Nm |

The mini uses 0.10 Nm hard leg caps for this regression; the medium retains its
4 Nm leg caps. Neither has settled leg torque clipping in these trials.
Roll/pitch kicks also pass. Both mini configurations pass standing-expression
tests; the optional tail passes a separate stationary wag test. Supported
animations are clearly labeled and are not used as balance evidence.

The [before-repair report](robot_design_2026_10/balance_before.json) records
41.1 mm fixed-tail drift and 77.7 mm tail-module drift over 30 seconds with the
previous gains. Both remained upright and met the old effort ceiling, so the
shorter test did not expose the eventual drift. A small gain screen on the
tail configuration selected lower gains, then both configurations and kick
cases were replayed. This is tuning of an ideal model, not motor identification.
The new CI horizon is 30 s for the mini; its settled cap-contact ceiling is
tightened from 60% to 10%. The drift bound was not relaxed.

Current full records:
[mini balance](../../examples/mini_tyrannosaurus_rex/balance_validation.json),
[mini engineering](../../examples/mini_tyrannosaurus_rex/validation.json),
[mini expression](../../examples/mini_tyrannosaurus_rex/character_validation.json),
[tail study](../../examples/mini_tyrannosaurus_rex/tail_validation.json), and
[medium validation](../../examples/tyrannosaurus_rex_50cm/validation.json).

## Open findings

1. **HIGH — Walking is unqualified.** With the revised mini gains, five of twelve
   nominal trials complete 30 seconds. At 0.20 Nm, the fixed and centered tails
   travel about 564/470 mm with 150/142 support alternations; wagging falls at
   3.592 seconds. All 0.10/0.15 Nm trials fall. Completed trials still encounter
   internal hip-link/motor-case contact and joint-limit overshoot. These are
   neither a collision-free gait nor evidence of sustained hardware capability.
   The medium has no demonstrated walking controller. Full observations are in
   [the walking report](../../examples/mini_tyrannosaurus_rex/walking_validation.json).
2. **HIGH — Hardware feasibility remains incomplete.** Assigned inertias,
   package boxes and ideal servos do not qualify real bearings, cable travel,
   manufactured shell mass, torque-speed/thermal behavior, supply limits or
   estimator delay. Measure those before procurement/fabrication acceptance.

These findings are also in [KNOWN_ISSUES.md](../KNOWN_ISSUES.md). The next
engineering work is a constrained weight-transfer/step test, followed by
contact-based gait acceptance, as proposed in
[ROBOT_WALKING_VALIDATION.md](../hardware/ROBOT_WALKING_VALIDATION.md).

## Verification commands

From a checkout with the repository's test and development dependencies:

```bash
ruff check environments/ examples/mini_tyrannosaurus_rex/ examples/tyrannosaurus_rex_50cm/ docs/hardware/studies/
ruff format --check environments/ examples/mini_tyrannosaurus_rex/ examples/tyrannosaurus_rex_50cm/ docs/hardware/studies/
mypy environments/ examples/mini_tyrannosaurus_rex/ examples/tyrannosaurus_rex_50cm/ docs/hardware/studies/ --ignore-missing-imports
MUJOCO_GL=egl pytest
python -m environments.shared.plant_contract --check
python -m environments.shared.species_catalog --check
python -m examples.mini_tyrannosaurus_rex.balance_checks --duration 30
python -m examples.tyrannosaurus_rex_50cm.validate --duration 30
```

CI supplies OSMesa for offscreen rendering and tests Python 3.11–3.13, plus the
SB3/torch typing and integration job. The local rendering check uses EGL. Read
the PR checks for the final tested commit; an earlier green check does not apply
to a later edit. Passing these tests establishes the stated simulated conditions,
not unsupported physical balance or a general recovery guarantee.
