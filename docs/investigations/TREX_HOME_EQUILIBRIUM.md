# T-Rex home-equilibrium investigation

> **Date:** 2026-07-23

## Decision

Repair the lower-body support mechanics before starting another PPO run. The
failure was not primarily a tail, neck, or root keyframe problem: the old
stance was an impact followed by an underpowered inverted-pendulum fall.

Once the physical stance had a robust basin, complete the policy interface in
the same unreleased plant revision: use the named-home residual mapping in
both Gymnasium and MJX, and make the foot sensors observe the plantar pads
that actually carry the load.

## Reproduction on the old plant

Under the Stage-1 configuration and the XML `home` command:

- the noise-free episode terminated as a nosedive at step 77;
- none of 50 resets at noise 0.05 survived the 1,000-step horizon;
- whole-body COM was at x=0.0866 m, while the loaded rear contact was near
  x=0.030 m;
- the root z=0.90 reset put the metatarsus surface about 90 mm below the
  floor and the toe surfaces about 75–78 mm below it;
- the leg default declared stiffness 40 but no stance `springref`, so the
  compiled springs pulled hip, knee, ankle, and toe coordinates toward zero
  while the controls pulled toward the keyframe; and
- the ankle moved from 90° toward its 50° lower limit while the torso pitched
  forward. The forces remained well below their bounds, identifying
  insufficient local servo stiffness rather than clipping.

The 85 kg T-Rex plant is roughly six times the raptor's mass, but its
hip/knee/ankle gains were only 1.3–1.5 times the raptor values.

## Rejected isolated fixes

Each candidate was compiled from a temporary XML and exercised through the
real Stage-1 environment:

- root pitch in either direction;
- hip-reference or keyframe-angle shifts;
- tail `springref`, tail stiffness, and tail-mass sweeps;
- neck and head stiffness;
- stronger gait gains without repairing the support contact;
- explicit leg `springref` without repairing support;
- fore-aft hip attachment and tail-mass shifts;
- random standing poses and symmetric constant-control searches.

The best one-dimensional balance shifts merely moved the plant across a
razor-thin boundary between nosedive and tail contact. They did not create a
robust basin. A thin support surface alone also failed because the original
servos let the stance joints sag before the front edge could carry load.

## Corrected stance contract

The final change keeps the authored physical leg pose while correcting how it
is represented and loaded:

1. Root/default and keyframe height are 0.9795 m, producing a shallow 0.5 mm
   loaded contact instead of a reset impact.
2. Each metatarsus radius is reduced from 0.04 to 0.025 m.
3. Each foot has a thin plantar box spanning the three toe bases and tips,
   creating a fore-aft support surface.
4. Every leg spring explicitly references its stance coordinate.
5. Knee, ankle, and toe joint references place the lower-body home controls at
   their range midpoints. The ankle retains a 5.5° target offset as gravity
   preload.
6. Hip-pitch, knee, and ankle gains use mass scaling plus the tested stance
   margin (1200/1500/900), with their existing 1.5×kp force headroom. Leg
   damping rises from 15 to 45; passive stiffness remains 40 so the policy can
   still articulate the legs.
7. The piecewise named-home residual mapping makes action zero command all 21
   home controls in Gymnasium and MJX while preserving the ±1 endpoints.
8. Box-shaped touch sites now share each plantar pad's metatarsus body and
   enclose its floor contacts. Four adjacent-toe exclusions remove the former
   28 mm d2/d3 and d3/d4 overlaps and their phantom self-contact forces.

## Acceptance results

All probes used the Stage-1 nosedive threshold of 0.35.

| Probe | Result |
|---|---:|
| Noise-free neutral action, 1,000 steps | 50/50 survive |
| JAX-style reset, 0.05 rad joint noise, nominal root height/velocity | 50/50 survive |
| Full SB3 reset, 0.05 joint/velocity/height noise | 47/50 survive |
| Raptor reference under the same full SB3 probe | 48/50 survive |
| Peak actuator force fraction in the full SB3 probe | 48.1% |
| Settled mean `forward_z`, noise-free | -0.048 |
| Exact-home plantar penetration | 0.5 mm |
| Settled foot touch sensors | ~419 N per foot |
| Airborne foot touch sensors | 0 N |
| Gymnasium ↔ MJX residual mapping/reset/contact parity | Pass |

The full SB3 misses are the three seeds whose independent Gaussian root-height
noise starts the long T-Rex legs about 8–11 cm inside the floor. JAX training
does not add root-height or velocity noise, and the JAX-style probe survives
all seeds.

## Remaining watch item

Re-measure gait excitation after any further gain, force-range, or reset
change. The piecewise residual map has unequal slopes only for T-Rex
neck/head controls because its lower-body home controls are midpoint-aligned;
watch early exploration for a persistent head/neck command bias.
