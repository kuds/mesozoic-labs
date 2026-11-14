# T. rex form, size and posture study

A design study for the Mesozoic Labs robotic *T. rex*. Round one compared four sizes (60, 75, 100 and
120 cm crown height), three widths, three body families and three postures, with real
component envelopes, load-based actuator sizing, power, cost and mechanical screening. Round two
develops the owner's choices (60 and 75 cm, Tapered and Hybrid bodies, 10° chest-up, Raspberry Pi 5
compute) with single-change variants and adds turning, carpet and behavior-envelope screens. Both
rounds are **design review packages**. Neither selects a final robot.

Stage 1 (8–10 Nov 2026) probed the three picks in closed loop in simulation: reference walks, a balance tracker fitted
by CMA-ES and small PPO runs. Stage 1b (10 Nov) refitted the tracker for speed and added servo delay. Stage 2 (from 10
Nov) aims at character with some agility: a lifelike reference walk, an expressive head and, next, a learned controller
trained on Colab. Every result is simulation. Nothing is measured on hardware and nothing is adopted without the owner
(P24).

| Read first | What it is |
|---|---|
| [LESSONS_LEARNED.md](LESSONS_LEARNED.md) | Where the work stands, what the robot can and cannot do in simulation, method and operating lessons, traps, next steps (14 Nov 2026) |
| [NEXT_SESSION_PROMPT.md](NEXT_SESSION_PROMPT.md) | The prompt that starts the next working session |
| [JETSON_PLEM_REVIEW.md](JETSON_PLEM_REVIEW.md) | Review of PR #668 (the 75 cm build with a Jetson Orin Nano Super): what to keep, what to treat with care, the next session's 75 cm + Jetson track and its owner decisions |
| [STAGE2_PLAN.md](STAGE2_PLAN.md) | Stage 2, the agility round with character: the agility bar, the learned controller R5, Colab training; owner decisions P28–P29 |
| [stage1/runs/2026-11-10-speed/STAGE1B_SPEED.md](stage1/runs/2026-11-10-speed/STAGE1B_SPEED.md) | Stage 1b report: speed with today's drives, a faster knee, servo delay |
| [stage1/runs/2026-11-08/STAGE1_PROBE.md](stage1/runs/2026-11-08/STAGE1_PROBE.md) | Stage 1 report: the closed-loop probe of the three picks, with the 60 cm gap map (corrected 10 Nov) |
| [STAGE1_PLAN.md](STAGE1_PLAN.md), [STAGE1B_PLAN.md](STAGE1B_PLAN.md) | Stage 1 and 1b plans with their numbered decisions |
| `stage2/look/`, `stage2/x/`, `stage2/research/` | Stage 2 so far: the look ladder's screen and clips, the expressive head (`-X`), the research notes |
| [tools/README.md](tools/README.md) | Environment setup, restoring the git-ignored inputs in a fresh clone, keep-alive and page tools |
| [ROUND2_REVIEW.md](ROUND2_REVIEW.md) | Round-two findings with the overnight refinement: the recommended bodies, the scorecard, the size study, materials and power, and the owner's questions |
| [round2/refine/MATERIALS.md](round2/refine/MATERIALS.md) | Materials (structure, coverings, foot pads) and power (battery, tether, the 3–4 ft robot), sourced and simulated |
| [ROUND1_REVIEW.md](ROUND1_REVIEW.md) | Round-one findings, comparison plates and the preference questions |
| [COMPUTE.md](COMPUTE.md) | Onboard compute survey (owner request P9) and the round-two compute budget |
| [BASELINE.md](BASELINE.md) | Source and evidence reconciliation: PR stack, missing R2/R3 handoff, uploaded posture study, exact baseline |
| [DEFINITIONS.md](DEFINITIONS.md) | How every height, width, ratio, posture angle, load and limit is defined |
| [DECISION_LOG.md](DECISION_LOG.md) | Preferences (P), engineering constraints (E), findings (F) and assumptions (A) |
| [BEHAVIOR_SPEC.md](BEHAVIOR_SPEC.md) | Proposed locomotion and autonomy targets for the selected design (not yet tested) |
| [REFERENCE.md](REFERENCE.md) | Anatomical references, how they map to the template, and licensing |
| [SCREENING.md](SCREENING.md) | Mechanical, clearance and sensor screening, including failed cases |
| [REPRODUCTION.md](REPRODUCTION.md) | Clean-checkout reproduction audit of PR #662/#663 |
| `r0_posture_study_2026-11-07/` | The uploaded 60 cm posture study, preserved with provenance |

## Layout

```
designlab/            the generator and screens (Python package)
  params.py           candidate IDs, height classes, widths, postures, proportion variants
  template.py         skeletal template and body-family shape tables (units of hip height)
  skeleton.py         world landmarks for a candidate: posture solve, neck, tail, legs
  hardware.py         component items, structure beam rules
  build.py            bodies, hardware placement, coverings -> Design
  mjcf.py             MJCF writer, exact thin-shell mass properties, ledger rows
  statics.py          stance IK, contact-consistent loads, single-support hold search
  gait.py             reference walk (LIPM/ZMP) + full inverse dynamics
  sizing.py           load requirements and actuator selection
  power.py, cost.py   runtime and prototype cost
  metrics.py          dimensions and ratios measured on the compiled model
  clearance.py        self-collision, swing and tail clearance, battery service path
  sensors.py          camera / ToF / lidar coverage by ray casting
  render.py, plates.py   review renders and plates (MuJoCo, offscreen)
  stage_a.py          the broad sweep;  round1.py  the review set, its screens and plates
  report.py           generated tables (round1/TABLES.md) and page data (round1/review_data.json)
  review_page.py, review_content.py, page_kit.py   the review web page
  turning.py          round two: prescribed turns, inverse dynamics, friction and clearance on every turn
  carpet.py           round two: pile model, toe catch, friction and a servo-held standing check
  envelope.py         round two: behavior envelope (animal-like walk, one-foot stop, push recovery, heat)
  round2.py           round two: sizing with the packaging search, every screen, plates
  report2.py          round-two tables (round2/TABLES.md) and data (round2/review_data.json)
  shortlist.py        round-two scorecard (version 2) and the bodies proposed for verification (round2/SHORTLIST.md)
  whatif.py, sizestudy.py, refine.py   75 cm what-if, size study, overnight refinement (levers, materials, power)
  gaittune.py         the animal-like walk tuned within fixed drives (F25)
  stage1/             Stage 1 and 1b: plant (servo envelope, delay, linkage), estimator, tracker (control.py), gym
                      env, CMA-ES fit (tune.py), speed fit (speedfit.py), PPO (train.py), panels, margin ledger, gates,
                      report; python -m designlab.stage1 --help
  stage2/             Stage 2: look ladder (look.py), Stage 0 screen (screen.py), clips (clips.py), clip metrics
                      (motion.py), look review page (look_page.py)
  data/               catalogs + raw sourced research records (URLs, access dates)
stage1/runs/          Stage 1 (2026-11-08) and Stage 1b (2026-11-10-speed) records, tables and reports
stage2/               Stage 2: look/ (ladder records, screen, clips), x/ (the expressive head), research/
tests/                pytest suite (conftest.py sets MUJOCO_GL=disable; see LESSONS_LEARNED.md §4.5)
tools/                environment setup, input restore, keep-alive, memory guard, report page and screenshot tools
round1/
  stage_a/            one compact JSON per candidate + summary.csv/json
  review/             review set: mjcf/ (ledgers; reference-size MJCF), screening JSON and PNG, plates
  TABLES.md           every comparison table, generated
round2/
  candidates/         one sized record per round-two candidate (20)
  review/             mjcf/ ledgers (the MJCF regenerates), screening JSON and PNG, plates
  TABLES.md, SHORTLIST.md, shortlist.json, review_data.json   generated
  CROSSCHECK.md/.json independent blind cross-check of the picks
  whatif_fr05/, size_study/, refine/   overnight: the what-if, the size study (SIZE_STUDY.md) and the refinement
                      experiments (REFINE.md, MATERIALS.md, fixed_drives.json, carpet_lift*.json, plates/)
```

## Reproduce

Rounds one and two need Python ≥ 3.11 with MuJoCo 3.10.0 (the repository's canonical version), numpy, scipy,
pillow and matplotlib. Rendering needs an offscreen OpenGL (OSMesa: `apt-get install libosmesa6`, or EGL on a GPU).
Stage 1 onward was run on Python 3.13.18 with the versions pinned in `tools/requirements-study.txt` (torch,
stable-baselines3 and gymnasium included); `tools/setup_venv.sh` builds that environment. Stage 1's PPO training and
its R3/R4 evaluation rows used a second set (torch 2.9.0+cpu, stable-baselines3 2.9.0, gymnasium 1.3.0):
`tools/requirements-stage1-ppo.txt`. A fresh clone lacks the
git-ignored model files and arrays the Stage 1 and 2 records hash: restore them with `tools/restore_inputs.py`
before running anything from Stage 1 on (`tools/README.md`). Run tests and training with `MUJOCO_GL=disable`, and
render only in a separate process (`LESSONS_LEARNED.md` §4.5).

```bash
bash hardware/trex_form_study/tools/setup_venv.sh
```

```bash
cd hardware/trex_form_study && . .venv/bin/activate
```

The commands below, from round one to the overnight refinement, run from `hardware/trex_form_study`. Stage 1, 1b and 2
commands are in their plans and run scripts (`STAGE1_PLAN.md`, `stage1/runs/2026-11-10-speed/run_speed_round.sh`,
`stage2/x/run_x.sh`, `python -m designlab.stage1 --help`, `python -m designlab.stage2.look --help`).

```bash
python -m designlab.cli build 100-M-S-10 --out /tmp/models
```

```bash
python -m designlab.cli size 100-M-S-10 --out /tmp/models
```

```bash
python -m designlab.stage_a --out round1/stage_a --xml-dir /tmp/stage_a_models --workers 3
```

```bash
python -m designlab.round1 --out round1/review
```

```bash
python -m designlab.report --stage-a round1/stage_a --review round1/review --out round1
```

```bash
python -m designlab.review_page --round1 round1 --out /tmp/review_page
```

Round two:

```bash
python -m designlab.round2 --out round2 --workers 3 --r1-models /tmp/r1_models
```

```bash
python -m designlab.report2 --round2 round2 --round1 round1 --out round2
```

```bash
python -m designlab.shortlist --round2 round2
```

```bash
python -m designlab.review2_page --round2 round2 --out /tmp/review2_page
```

Overnight refinement (each writes its own directory under `round2/`; `--resume` keeps finished bodies):

```bash
python -m designlab.sizestudy --out round2/size_study --workers 3
```

```bash
python -m designlab.refine --out round2/refine --workers 3 --cases carpet_60 targeted_75 knee_60 knee_75
```

```bash
python -m designlab.refine --out round2/refine --fixed --cases
```

```bash
python -m designlab.refine --out round2/refine --cases --carpet-lift --lift-cases targeted_75 --lifts 0.175 0.18
```

```bash
python -m designlab.refine --out round2/refine --cases --carpet-lift --lift-cases narrow_carpet_60 --lifts 0.19 0.195 --nominal-only
```

```bash
python -m designlab.refine --out round2/refine --plates-picks --report-only
```

```bash
R=round2/refine; python -m designlab.shortlist --round2 round2 --extra round2/whatif_fr05 $R/carpet_60 $R/targeted_75 $R/narrow_carpet_60 $R/feet_60 $R/feet_75 $R/knee_60 $R/knee_75 $R/no_calf $R/pads_large $R/toes_long $R/foot_both $R/narrow_3mm $R/headroom_60
```

The carpet swing pass (`--carpet-lift`) regenerates the reference walks with each lift and checks them against
the selected drives and the walking clearance gates; `--nominal-only` lifts the nominal walk alone, the carpet
gait's speed. `--plates-picks` draws round two against the recommended bodies.

`round2` sizes the 20 round-two candidates (with the packaging search), rebuilds each one (SHA-256
checked against its sized record) and runs the clearance, sensor, turning, carpet and behavior-envelope
screens, then draws the plates: about an hour on 3 cores. Its reference walks run at round two's accuracy
(`gait.set_accuracy("r2")`, decision log F14); the default keeps round one reproducible. `report2` and
`shortlist` only format and rank recorded results. `python -m designlab.turning <id> --round2 round2 --out <dir>`
and `python -m designlab.envelope round2 <id>` screen single round-two candidates; `python -m designlab.carpet <id>
--out <dir>` screens round-one (Stage A) bodies only, at round one's walk accuracy.

`build` writes one candidate's MJCF and ledger. `size` adds load-based actuator and battery
selection. `stage_a` runs the full sweep, about an hour on 3 cores. `round1` rebuilds the review set
from the Stage A selections (each model's SHA-256 is checked against the sweep record), then runs the
clearance and sensor screens and draws every plate. `report` writes the tables, and `review_page`
writes the review page with web-sized images. `python -m designlab.clearance <id>` and
`python -m designlab.sensors <id>` screen a single candidate.

Candidate IDs read `<crown cm>-<width N|M|W>-<family T|S|H>-<posture 00|10|20>`, with optional
suffixes `-anat`, `-compact`, `-bighead` (proportions), `-side`, `-serial` (leg layout) and `-belly`
(battery position). Example: `100-M-H-10`. Every generated MJCF opens in MuJoCo's viewer
(`python -m mujoco.viewer --mjcf=<file>`) at its `home` keyframe.

## What is and is not claimed

* Masses, inertias, loads, powers, runtimes and costs come from models built from sourced part data and stated rules. **None is measured.** Every ledger row carries its basis (`sourced` / `derived` / `estimate`).
* The reference walk is a prescribed trajectory for sizing. It is not a controller and does not show that any controller can walk the robot.
* Stage 1 and 1b controllers (the fitted tracker and small PPO runs) are simulation probes of the picks, with ideal servos behind a torque-speed envelope and a modelled delay. Their verdicts are simulation evidence about those bodies and controllers, not hardware results, and no controller has been chosen for the robot.
* Renders show the generated models themselves. No concept art is used.
* Physical construction, simulation evidence and hardware measurements are kept separate. Rounds one and two contain only the first two.
* The turning, carpet and behavior-envelope screens check prescribed motions and simple balance models against the selected drives with the sizing margins. Passing them does not show that a controller can produce the behavior; failing one shows that the prescribed motion exceeds that body and those drives at those margins, not that no controller can.
