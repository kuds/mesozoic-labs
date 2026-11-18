# Pre-registration: statically stable scripted crawl (task 4)

Written 18 Nov 2026, **before** any crawl was run. Rules and seeds are fixed here; any rule changed after a result will
be labelled "changed after a result". The crawl's gait parameters are tuned on development seeds (below), then frozen
in a commit before the judged seeds run.

## What is run

- **Models:** `generated/trike_T20_closed.xml` (20.00 kg) and `generated/trike_T15_closed.xml` (15.00 kg), as
  committed in dc2e405 or as corrected after the independent model check (any correction is listed in the report).
- **Controller:** `crawl.py`, a scripted, open-loop crawl. One foot moves at a time in the order left hind, left fore,
  right hind, right fore. Before each foot lifts, with all four feet down, the body moves (forward and sideways) so
  that the planned centre of mass lies inside the triangle of the three feet that stay down, with the **planned margin
  ≥ 30 mm**. The plan computes the centre of mass from the full model at both the start and the end of the swing. Joint
  targets come from inverse kinematics of the planned foot and body positions; the drives are the model's position
  servos. No feedback from the simulated body pose.
- **Margin basis:** support at the ankle pins (point feet), the conservative basis. The margin with each sole's 80 mm
  width counted at its pin is reported too, not judged.
- **Run:** 1 s of standing, then 8 full gait cycles (32 single-foot steps), then 1 s of standing.
- **Judged seeds:** 1 and 2, for each build. Each seed draws: a start offset on every hinge joint, uniform in
  ±0.02 rad; and one sliding friction coefficient for the floor and the soles, uniform in [0.6, 1.0]
  (`numpy.random.default_rng(seed)`, joint offsets drawn first).
- **Development seeds:** 100 and up. Tuning uses only these. Budget: at most 60 development runs; stop rule: if no
  parameter set passes every rule on development seeds 100-103 for both builds within that budget, stop and report the
  crawl as not achieved, with the best attempt.
- **Sampling:** every 10 ms.

## Rules

| Rule | What | Pass if |
|---|---|---|
| C1 | Upright | Trunk tilt ≤ 10° and chassis height ≥ 0.85 × its height after the first second, at every sample; no NaN; no MuJoCo warning |
| C2 | Progress | Forward travel of the chassis over the run ≥ 80 % of the planned 8 strides; sideways drift ≤ 0.10 m; heading change ≤ 10° |
| C3 | Static margin, measured | At every sample in which a foot is in its swing window, the centre of mass's ground projection lies inside the triangle of the three planted feet's ankle pins with **margin ≥ 10 mm** |
| C4 | One foot at a time | For every swing, the swinging foot carries < 5 % of the weight for ≥ 80 % of its swing window, and each of the other three feet carries ≥ 5 % of the weight throughout that window |
| C5 | Planted feet hold | Each foot's ankle pin moves ≤ 10 mm in the ground plane between its touchdown and its next lift-off (from 0.10 s after touchdown) |
| C6 | Drive limits | No actuator at ≥ 99 % of its torque limit in more than 1 % of samples |

**Verdict per build:** PASS if C1-C6 hold on both judged seeds. Reported but not judged: speed, the margin on the
sole-width basis, each joint's peak and RMS torque and peak speed (judged later, in task 7, against the parts list and
the drives' limits), and the centre of pressure across each sole.

## What this does not show

A crawl is not the 0.30 m/s museum walk. It shows that the body can be shifted inside the support triangle and one foot
lifted at a time with a positive static margin, in simulation, without delay, noise or sensor models (task 5 adds them).
