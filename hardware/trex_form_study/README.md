# T. rex form, size and posture study

A design study for the Mesozoic Labs robotic *T. rex*. Round one compared four sizes (60, 75, 100 and
120 cm crown height), three widths, three body families and three postures, with real
component envelopes, load-based actuator sizing, power, cost and mechanical screening. Round two
develops the owner's choices (60 and 75 cm, Tapered and Hybrid bodies, 10° chest-up, Raspberry Pi 5
compute) with single-change variants and adds turning, carpet and behavior-envelope screens. Both
rounds are **design review packages**. Neither selects a final robot, and no controller has been trained.

| Read first | What it is |
|---|---|
| [ROUND2_REVIEW.md](ROUND2_REVIEW.md) | Round-two findings, the scorecard, the three bodies proposed for verification and the round-two questions |
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
  shortlist.py        round-two scorecard and the bodies proposed for verification (round2/SHORTLIST.md)
  data/               catalogs + raw sourced research records (URLs, access dates)
round1/
  stage_a/            one compact JSON per candidate + summary.csv/json
  review/             review set: mjcf/ (ledgers; reference-size MJCF), screening JSON and PNG, plates
  TABLES.md           every comparison table, generated
round2/
  candidates/         one sized record per round-two candidate (20)
  review/             mjcf/ ledgers (the MJCF regenerates), screening JSON and PNG, plates
  TABLES.md, SHORTLIST.md, shortlist.json, review_data.json   generated
```

## Reproduce

Requires Python ≥ 3.11 and the packages below. MuJoCo 3.10.0 is the repository's canonical
version. Rendering needs an offscreen OpenGL (OSMesa: `apt-get install libosmesa6`, or EGL on
a GPU).

```bash
cd hardware/trex_form_study
python -m venv .venv
. .venv/bin/activate
pip install "mujoco==3.10.0" numpy scipy pillow matplotlib
```

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

`round2` sizes the 20 round-two candidates (with the packaging search), rebuilds each one (SHA-256
checked against its sized record) and runs the clearance, sensor, turning, carpet and behavior-envelope
screens, then draws the plates: about an hour on 3 cores. Its reference walks run at round two's accuracy
(`gait.set_accuracy("r2")`, decision log F14); the default keeps round one reproducible. `report2` and
`shortlist` only format and rank recorded results. `python -m designlab.turning`, `python -m designlab.carpet`
and `python -m designlab.envelope` screen single candidates.

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
* Renders show the generated models themselves. No concept art is used.
* Physical construction, simulation evidence and hardware measurements are kept separate. Rounds one and two contain only the first two.
* The turning, carpet and behavior-envelope screens are necessary-condition checks on prescribed motions and simple balance models. Passing them does not show that a controller can produce the behavior; failing one shows that no controller can with that body and those drives.
