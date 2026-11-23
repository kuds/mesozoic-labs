# Walking and steering on the 60 cm open-rib model

Run from this directory using the Python environment in [the refinement requirements](../requirements.txt). The commands write to a new directory named `trials` relative to the current directory. `--record` also produces a MuJoCo pose-and-load NPZ for playback.

```sh
python run.py stand --out trials/stand
python run.py walk --record --out trials/walk
python run.py course --record --out trials/course
python run.py pivot-right --record --out trials/pivot-right
python run.py walk --gait reference --model ../models/reference/physics.xml --out trials/reference
python panel.py --group all --jobs 4 --out trials/refined-panel
python panel.py --model ../models/reference/physics.xml --gait reference --jobs 4 --out trials/reference-panel
```

The default is the revised physical model and free-base four-beat walk. The original gait's fixed target file reproduces the reference controller exactly. The revised gait re-solves leg IK from the stored support waypoints, with plantar targets relative to each ankle. To isolate the effect of hardware placement, pair the revised model with `--gait reference`. To isolate the gait, pair the reference physics model with `--gait refined`.

The target stride is 96 mm on both walks. The revised walk reduces planned lift from 32 to 26 mm, scales the whole clock by 0.8 and takes one footfall every 1.073 s. It uses 0.748 of the original lateral support target after startup and removes the backward body offset during the first two setup swings. One separate neck glance takes place in each walk. The walk uses position servos. The moving heading course is a separate diagonal gait with 70% requested stance, a 0.25 m/s command and a 10 ms leg delay.

The straight-walk panel uses seeds 11–14 and 21–24 on nominal and stiffer floor contact. Use the self-contained `physics.xml` for `--stiff`. The inherited crawl judge tests upright posture, forward progress, support margin, contact sequence and actuator saturation. The C5c slide criterion tracks the actual point of contact as the capsule sole rolls. It allows up to 10 mm sliding from 0.10 s after each touchdown to lift-off. C3 projects the center of mass into the triangle defined by the three nonswing ankle positions during planned swing. The limit is 10 mm. The JSON also gives the older C5 ankle-travel measure. Ground-reaction peaks for walking are sampled at 1 kHz during the steady sequence following the first two setup swings.

Heading trials command 0°, +35°, −35°, and 0° over 23 s. The panel uses seeds 101–103 on nominal and randomized plants with sensor noise. A command is acquired when the simulated body stays within ±5° for a continuous second. Two separate 12 s zero-speed pivot trials command ±40°. To reproduce the gain sensitivity, run `panel.py --group steering --yaw-gain 1.0` or `2.0` with a fresh output directory. The adopted heading gain is 1.5 per second. Steering force measurements cover each 1 ms physics step after two seconds. The foot-load order is left fore, right fore, left hind and right hind.

The largest single-foot force in the matched trials occurred on the right forefoot during a stationary right pivot. For seed 102, both force peaks occur 0.631 s after the turn request. The instantaneous peaks are 571.88 N on the reference and 580.23 N on the revision. Their largest rolling 10 ms means are 404.02 and 399.82 N. Reproduce the [recorded comparison](../evidence/pivot-right-102.json) with:

```sh
python run.py pivot-right --seed 102 --model ../models/reference/physics.xml --out trials/pivot-ref --force-trace trials/pivot-ref.csv
python run.py pivot-right --seed 102 --out trials/pivot-revised --force-trace trials/pivot-revised.csv
```

The CSV contains 1 ms normal-force samples for all feet around both peaks. The raw peak and the mean of ten consecutive samples use the entire steering scoring interval. The 50 Hz animation frames cannot recover this impact peak.
