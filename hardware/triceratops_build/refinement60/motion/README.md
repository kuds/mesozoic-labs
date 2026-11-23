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

## Gentle turn on the same 60 cm hardware

Select `--turn-profile gentle` for the moving course or stationary pivots. Both profiles use the same 12.844 kg nominal model, gyro noise, ten milliseconds of leg actuation delay, 0.25 m/s moving speed request and command times. The original profile remains the default. The comparison videos use synchronized recorded poses from [the seed 102 stationary right pivot](../media/gentle-pivot-right.mp4) and [the seed 103 moving course](../media/gentle-course.mp4).

The original heading adapter allows a 0.40 rad/s yaw request and a step change. The gentle adapter caps the request at 0.34 rad/s and changes it by at most 0.65 rad/s². The 0.70 s stride clock and 1.5/s heading gain are common. Moving turns use 28 mm commanded foot lift and 70% stance. At zero forward speed, the gentle pivot uses 12 mm lift and 64% stance, and finishes the active swing before parking. The original uses 35 mm and 70% in both modes. Lower clearance suits the flat simulated floor.

The gentle adapter integrates measured body gyro rates projected onto the Euler yaw axis using the noisy gravity direction, starting from the robot's initial bearing. It receives the operator's requested relative heading. The course judge scores the simulated chassis separately. The simulated gyro carries 0.02 rad/s per-axis sample noise and zero constant bias by default. The original adapter uses a different observation model with a yaw random walk of 0.01 rad/√s. The gentle result therefore depends on the raw IMU and the short run duration.

| Panel and maneuver | Courses or pivots acquired, original → gentle | Highest raw foot load, N | Highest 10 ms mean, N | Worst loaded-contact slip p95, m/s |
|---|---:|---:|---:|---:|
| Seeds 101–103, moving | 4/6 → 6/6 | 354.8 → 309.4 | 233.7 → 224.3 | 0.072 → 0.078 |
| Seeds 101–103, pivot | 6/6 → 6/6 | 580.2 → 302.5 | 399.8 → 210.2 | 0.016 → 0.020 |
| Held-out 104–106, moving | 6/6 → 6/6 | 344.7 → 385.3 | 269.0 → 257.6 | 0.069 → 0.075 |
| Held-out 104–106, pivot | 6/6 → 6/6 | 379.6 → 301.1 | 331.2 → 217.6 | 0.016 → 0.022 |

Each set contains nominal and randomized moving courses on three seeds, plus one left and one right nominal pivot per seed. Each six-course set contains 24 scored segments including the launch. All 48 paired simulations finished upright. Per-foot loads, times, final bearings, slip, transient lateral offset, chassis tilt and actuator saturation are in [the paired trial data](gentle-turn-results.json).

- On the seed 102 right pivot, RF fell from 580.23 N at 3.631 s to 264.65 N at 4.301 s. Its worst ten-sample mean fell from 399.82 to 210.17 N. The sustained ±5° band began 1.83 → 2.47 s after the command. Peak chassis excursion from the command position was 22.8 → 20.3 mm. Final bearing was −38.92° → −37.81° for a −40° request.
- On seed 103 nominal, the old course ended at −6.16° and missed its final 0° hold. The gentle course entered the band at 17.90 s and ended at +0.52°. Across seeds 101–103 the longest successful moving acquisition grew from 3.36 to 3.98 s. Worst transient cross-track displacement from a commanded segment's starting tangent rose from 0.353 to 0.403 m. After turns settled, final-two-second projected speeds on the revised moving runs were 0.256–0.259 m/s.
- The highest gentle moving load across both panels was 385.33 N on the LF during randomized seed 106. The largest gentle pivot loads across both panels were 301.10 N LF, 302.47 N RF, 180.41 N LH and 174.16 N RH. The impact improvement varies by foot, seed and maneuver.
- In a diagnostic run, `--turn-profile shaped-stock-yaw` kept the original heading input and applied the gentle stepping and rate limits. Seed 102's RF pivot peak still fell to 264.65 N. Both seed 103 moving courses still missed the final heading. The revised heading estimate recovered those holds.

### Bias sensitivity

A separate perturbation added constant body-Z gyro bias at seeds 103 and 105, nominal and randomized, with the gentle controller. At ±0.10°/s all eight courses acquired every segment and ended within ±5° of straight. At ±0.25°/s six of eight acquired every segment, but all eight ended outside the band. At ±0.50°/s all eight missed at least one segment, with final heading errors of 10.8–11.7°. Entry for one continuous second and accuracy at the end of the run answer different questions. Check gyro bias against a measured floor bearing and repeat the turn under load before extending these results to longer demos. The 1 kHz force numbers describe the contact model and require instrumented hardware confirmation.

### Reproduce the paired panels

Run each profile in a separate directory. `--force-trace` writes the 1 kHz excerpts around the raw and rolling peaks for a single trial. `--record` includes full-episode `physics_t_s` and `physics_foot_load_N` arrays in steering NPZs, alongside the 50 Hz pose stream.

```sh
python panel.py --group steering --turn-profile original --steering-seeds 101 102 103 104 105 106 --jobs 4 --out trials/original
python panel.py --group steering --turn-profile gentle --steering-seeds 101 102 103 104 105 106 --jobs 4 --out trials/gentle
python run.py pivot-right --seed 102 --turn-profile gentle --record --force-trace trials/rf.csv --out trials/gentle-rf
python run.py course --seed 103 --turn-profile shaped-stock-yaw --out trials/shaped103
python run.py course --seed 105 --turn-profile gentle --gyro-bias-dps 0.25 --random --out trials/bias-plus025
```

## Six distinct requested turns

The 40 s moving program starts straight and then changes bearing six times, to +15°, +35°, +10°, −15°, −35° and 0°. Use `six-headings` through the same runner to reproduce the [twelve paired runs](../evidence/six-distinct-headings.json). The film follows nominal seed 103. Its force plot shows the right forefoot throughout. The evidence JSON reports maxima for all four feet.

```sh
for profile in original gentle
do
  for seed in 101 102 103
  do
    python run.py six-headings --seed "$seed" --turn-profile "$profile" --out "trials/six-$profile-$seed-nominal"
    python run.py six-headings --seed "$seed" --turn-profile "$profile" --random --out "trials/six-$profile-$seed-random"
  done
done
```
