# Tail posture decision: corrected summary (2026-10-25)

The draft brief (brief_draft.md) and two skeptics' findings (journal of workflow [id removed]) govern this summary. The skeptics found
four blocking problems in the draft's framing and recommendation; corrections applied below. Nothing trained.

## Measured (re-derived by the facts skeptic from raw traces; FK on a byte copy of the r9 MJCF)
- BOUND60 holds tail_3 at q3 = -2.28 deg (command about 0.62), steady within episodes (per-episode SD at most 0.0036 rad).
- Against the one cured control (s42 r202, n = 1, still descending at R2): BOUND60 is 1.7 deg higher at the joint and 2.1 deg at the
  distal segment over the R1/R2 mean (2.5 and 2.9 deg at R2; tip 27-37 mm).
- Against the walking-pilot parent (stance3 Run 2 final, q3 -1.09 deg, 0/40 flagged): BOUND60 is 1.2 deg LOWER at the joint.
- Rail controls and BOUND90 sit about 1.5 deg higher at the joint than BOUND60.
- In every arm the distal tail points 22-28 deg below horizontal and tail_1 sits on its ventral stop; r9 cannot hold a level tail.
- Cost: no falls at R1/R2, no other-actuator saturation, no OTHER flags in 1M steps; returns vs rail-class controls +36 to +43 (s42) and
  +19.8 (s80); vs the single cured control -18 to -23 at R2 (n = 1). K3 cannot be read in 1M steps.
- The outward drive persists under BOUND60: 52-59% of training rows have tail_3's mean above b = 0.60 [skeptic, telemetry].
## Corrections to the draft
- The PARKED rule was adopted so that a bound which only masks the outward drive is not credited (RUNLOG 16:00; kept literal 17:46 with
  the clean-holding-states facts already in front of the maintainer). It is not an anatomy rule. BOUND60's telemetry shows the drive
  persists, so on the rule's own purpose BOUND60 masks rather than cures.
- Anatomy does not rank the r9 postures: they differ by 1.5-3 deg inside a band 22-28 deg below the level tail reconstructions favour
  (soft literature: trunk horizontal, Newman 1970; CoM near the hip, Henderson 1999, Allen et al. 2013; no source fixes a tail angle).
- A level-tail servo design needs headroom: the break-even kp/ctrlrange values hold level only at a = +1 (the rail); passive support
  (springref) is the route that holds a level tail off the rail. Any r10 tail chain is a new lineage (every r9/r15 checkpoint stops loading).
- Missing options added: a CPU persistence screen of BOUND60 to the K3 slots; a lower bound (b about 0.45-0.50) with the masking caveat.
