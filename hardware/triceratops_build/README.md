# Triceratops build

The working folder for the four-legged Triceratops museum robot, started 18 Nov 2026. Simulation first: MuJoCo models,
then a walking controller in simulation, then a one-foreleg bench plan. **Nothing here is built, bought, quoted or
measured.** Simulation results are labelled as simulation; targets and estimates are labelled as such.

- **The robot (targets):** 75 cm to the top of the frill, 1.5 m long, 15 or 20 kg (owner's decision D1), walking
  slowly at 0.30 m/s (0.7 mph).
- **Museum dates (the owner's):** first contact mid-December 2026 with simulation renders; live demos in Q3 2027.
- **The T. rex** is paused and preserved, and stays the flagship for end 2027 / early 2028. This folder does not touch
  its files; where a choice here affects hardware the two robots share, the file says so.

## Where things are

| File | What it is |
|---|---|
| [DECISION_LOG.md](DECISION_LOG.md) | The owner's requirements and decisions (with the options put), constraints, findings (F), assumptions (A) |
| [pr669_review/](pr669_review/REVIEW.md) | Review of PR #669 (sound as a concept study), draft errata for its README and its PDF, checks run |
| [outreach/QUESTIONS_WIM_MYACTUATOR_DRAFT.md](outreach/QUESTIONS_WIM_MYACTUATOR_DRAFT.md) | Draft questions for WIM Robotics (PLEM) and MyActuator, for the owner to send |
| [sim/STAND_REPORT.md](sim/STAND_REPORT.md) | Task 3: the T20 and T15 models with the parts list's mass distribution; four-foot standing (all pass) |
| [sim/CRAWL_REPORT.md](sim/CRAWL_REPORT.md) | Task 4: scripted crawl; fails with PR #669's foot, passes with a passive roll ankle (design proposal) |
| [renders/](renders/README.md) | Task 9: labelled simulation stills for the December museum contact; clip script |

The simulation code is in [sim/](sim/): `build_model.py` (models, mass ledgers), `check_stand.py`, `crawl.py`,
`crawl_judged.py`, `crawl_extra.py`, `support_margin.py`; pre-registrations `PREREG_STAND.md`, `PREREG_CRAWL.md`;
results under `sim/results/`.

Inputs, in `hardware/triceratops_kickoff/`: the checked parts list (`PARTS_PREMIUM.md`), the consolidation plan with the
milestones and the open decisions (`CONSOLIDATION_PLAN.md`), the PLEM fit analysis (`PLEM_FIT.md`) and their reviews.
PR #669's study folder, `hardware/triceratops_design_study/`, stays as its record once merged.

## Reproducing the checks

```sh
python -m pip install mujoco==3.10.0 numpy
git fetch origin pull/669/head:pr-669
mkdir -p /tmp/pr669 && git archive pr-669 hardware/triceratops_design_study | tar -x -C /tmp/pr669
(cd /tmp/pr669/hardware/triceratops_design_study/models && python check_models.py)
python hardware/triceratops_build/pr669_review/stance_balance_check.py \
  /tmp/pr669/hardware/triceratops_design_study/models/robot.xml
```

## Plan (from the kickoff prompt)

1. Review PR #669; draft its errata. *Done 18 Nov 2026, independently checked twice; the owner applies and merges.*
2. Owner decisions, round 1 by 15 Dec 2026 (mass, size, compute, bus voltage, chassis) and round 2 by 15 Jan 2027
   (T. rex milestone, PLEM pilot, pull-request triage). Draft the WIM and MyActuator questions. *Questions drafted;
   round 1 put on 18 Nov 2026 and declined, so all decisions are open.*
3. MuJoCo standing model of the chosen build with its real mass distribution, drive limits and transmissions.
   *Done for both T20 and T15 (simulation; all pass).*
4. A statically stable scripted crawl, margin pre-registered. *Done: fails with PR #669's foot, passes with a
   passive roll ankle (simulation; proposal for the owner).*
5. A 0.30 m/s walk with 0-10 ms servo delay and noise, on grippy and slippery floors; two controllers.
6. A short trot at 0.50-0.60 m/s.
7. Actuator sizing confirmed in simulation against the parts list and each drive's limits.
8. A one-foreleg bench plan with pass and fail rules written first.
9. Museum renders by 8 Dec 2026 (start by 1 Dec 2026), each labelled "simulation". *Stills done 18 Nov 2026; the
   clip waits for the owner's choice of where clips go.*

## Next steps (18 Nov 2026)

**For the owner**
- PR #669: paste `pr669_review/ERRATA_README_SECTION.md` into its README, send `ERRATA_PDF_FOR_MAINTAINER.md` to the
  PDF's maintainer, merge with a merge commit.
- Round 1 decisions by 15 Dec 2026 (all still open; the first ask was declined): mass (D1), size (D2), compute (D3),
  bus voltage (D4), chassis (D5), now including **the foot**: a sole that stays flat in roll is needed (F7); a passive
  roll hinge is the first candidate but is not yet robust in simulation.
- Send the drafted questions in `outreach/` to WIM Robotics and MyActuator (and the T. rex's five maker questions, if
  not sent), and tell the session the date.

**For the session (decided 18 Nov 2026: D9, D10)**
- Task 5, the 0.30 m/s walk: a model-based controller first, in this container (CPU); the learned policy on Colab
  later, after its own cost estimate. Verdict judged at 10 ms of servo delay. Pre-register before any judged run. Port
  the T. rex delay queue (`designlab/stage1/speedfit.py`, `servo_delay`) and the parts of its servo model
  (`designlab/stage2/servo.py`) that apply; check every result under at least one stiffer contact setting (the crawl's
  lesson); measure sole slip and tilt separately from pin motion.
- The foot: try a roll hinge whose spring centres the sole to the ground rather than to the leg, and re-test the crawl
  under both contact settings.
- Model: an angle-dependent ratio for the 1.3:1 pushrod (A5).
- Renders: redo before 8 Dec 2026 if round-1 decisions change the build; walking clips go on the data branch
  `claude/triceratops-build-clips`.
