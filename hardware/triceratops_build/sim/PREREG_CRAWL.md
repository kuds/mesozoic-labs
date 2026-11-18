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

## Amendment 1 (18 Nov 2026, after 48 development runs, before any judged run)

Written before seeds 1 and 2 were run on any model. The rules C1-C6, the seeds and the sampling are unchanged.

**Changed during development (labelled as such; no judged result existed):**
- **Gait start and end.** Two setup swings (right fore half a stride ahead, right hind a quarter) put the feet into the
  steady stagger before the 8 cycles, so a run has 34 swings, not 32; a final four-foot shift returns the body to the
  centreline and the middle of the feet before the last second of standing. C2's planned travel stays 8 strides.
- **Foot placement.** Each cycle swing lands half a stride ahead of its hip on the body's steady path.
- **Contact preload.** Planted feet are commanded 2 mm below the floor.
- **Frozen parameters** (`crawl.py`, `PARAMS`): stride 0.12 m, lift 40 mm, shift 1.0 s, swing 0.5 s (20 % lifting and
  20 % lowering), planned margin 65 mm (the rule asks ≥ 30 mm), sideways body shift ≤ 100 mm, fore-aft adjustment
  ≤ 80 mm with sideways shift weighted 3:1 against it. Planned speed 0.12 m / 6 s = 0.02 m/s.
- **Models:** corrected after the independent model check (reserve placement, drive collision twins, leg gains from
  the T. rex servo model's nominal); see `STAND_REPORT.md`.

**Stop-rule outcome for the models as pre-registered (PR #669's ankle: passive pitch only, sole rigid in roll).** No
parameter set passed every rule on development seeds 100-103 within the 48 runs used. C4 (the third foot loaded) and
C5 (planted ankle pins hold) conflict for this foot: the two diagonal soles carry the robot on their edges, so the third
foot floats; and any sideways body shift leans the legs, which tips the roll-rigid soles onto their edges and moves the
pins. Details: `CRAWL_REPORT.md`. The judged seeds are still run on these models with the frozen parameters, so they
get a formal verdict.

**Added: a design-variant experiment, pre-registered here.** Models `generated/trike_{T20,T15}_closed_ankleroll.xml`
(`build_model.py --ankle-roll`): the same robots with a passive roll hinge added at each ankle (±20°, the same spring,
damping and armature as PR #669's passive pitch ankle). Same controller, frozen parameters, rules, seeds and verdict
logic as above. Development seeds 100-103 passed every rule on both builds (runs 39-48). This is a design proposal for
the owner (chassis decision D5), not an adopted change.

**Judged runs:** seeds 1 and 2 on the four models `trike_T20_closed`, `trike_T15_closed`, `trike_T20_closed_ankleroll`,
`trike_T15_closed_ankleroll`, by `crawl_judged.py`, results in `results/crawl_2026-11-18/`.
