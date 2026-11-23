# Six-heading speed sweep with the diagonal gait

Run 40 s on the 60 cm open-rib chassis using the existing gentle steering controller. After 3 s straight, the requested bearings are +15°, +35°, +10°, −15°, −35° and 0°. The setup and heading deadlines stay fixed at every speed. Both models use the same 0.70 s diagonal-pair stride, 70% commanded stance, 28 mm foot lift, 10 ms motor-command delay, gyro yaw estimate, torque limits and level floor. Seeds 101–103 each supply a nominal and a randomized run at every requested speed. Seeds 104–106 independently repeat selected points on both plant settings. [Main results](steering-main.json) · [Additional-seed results](steering-holdout.json)

The existing command input goes to 0.30 m/s. Higher requests expand the simulation's speed-input normalization. The controller receives forward velocity in physical units. Its gains, footfall clock, foot placement algorithm and motor limits stay fixed. The tested speeds include a stress region that trips the upright monitor.

A heading counts after staying within ±5° of true chassis yaw for a continuous second. Each completed segment's final bearing error is also recorded. The simulated controller sees its IMU observations and requested relative heading. Speed is computed from progress projected along each requested bearing, weighted by time and including launch and turns. The table averages achieved speed over completed trials. Early-stopped trials retain their individual achieved speed and stop time in the evidence. Force is the foot contact normal reported by the existing touch sensors, sampled at 1 kHz after the first 2 s. Each rolling value averages ten consecutive samples on one foot. A tilt-stop trial measures impact up to its stop.

## Matched results

Each cell gives **128 mm / 160 mm**. A complete course requires all six 1 s heading holds and a full upright 40 s run. Loads and flight include all six trials per geometry, including the observed portions of tilt-stopped trials.

| Command (m/s) | Mean actual m/s, completed runs | Complete courses out of 6 | Worst raw load (N) | Worst 10 ms mean (N) | Worst airborne time (%) |
|---:|---:|---:|---:|---:|---:|
| 0.100 | 0.097 / 0.097 | 6 / 6 | 491 / 392 | 368 / 320 | 0.03 / 0.03 |
| 0.150 | 0.148 / 0.147 | 6 / 6 | 415 / 316 | 237 / 221 | 0.03 / 0.05 |
| 0.200 | 0.198 / 0.196 | 6 / 6 | 417 / 372 | 252 / 309 | 0.03 / 0.04 |
| 0.250 | 0.246 / 0.245 | 6 / 6 | 416 / 281 | 261 / 215 | 0.01 / 0.00 |
| 0.300 | 0.295 / 0.294 | 6 / 6 | 251 / 201 | 214 / 186 | 0.00 / 0.00 |
| 0.350 | 0.346 / 0.343 | 6 / 6 | 203 / 195 | 183 / 187 | 0.01 / 0.02 |
| 0.400 | 0.399 / 0.397 | 6 / 6 | 346 / 247 | 273 / 229 | 0.01 / 0.00 |
| 0.450 | 0.449 / 0.447 | 6 / 6 | 445 / 321 | 374 / 306 | 0.01 / 0.00 |
| 0.500 | 0.470 / 0.458 | 6 / 6 | 782 / 601 | 585 / 483 | 1.73 / 2.10 |
| 0.550 | 0.472 / 0.466 | 6 / 4 | 771 / 639 | 586 / 543 | 6.03 / 5.81 |
| 0.575 | 0.478 / 0.384 | 3 / 1 | 772 / 1024 | 563 / 522 | 11.09 / 13.09 |
| 0.600 | — / — | 0 / 0 | 1007 / 949 | 566 / 495 | 14.28 / 18.76 |

All twelve additional-seed trials per build at 0.45 and 0.50 m/s (six per speed) completed with all headings. At 0.55 m/s, six of six additional 128 mm courses and three of six additional 160 mm courses met the same requirement. Across both seed sets that gives 12/12 and 7/12 respectively. On the primary panel, all twelve 0.60 m/s attempts terminated at the model’s tilt stop.

A small time with no loaded feet at moderate speed can come from brief load transitions; the underlying threshold is 2 N per foot at 1 kHz. The simultaneous rises in airborne time, grounded-contact slip and forefoot impulse indicate a distinct change in the fast gait above 0.45 m/s. Heading success alone can conceal this degradation. This is a fixed diagonal footfall gait, distinct from the slow four-beat exhibit walk.

## Reproduce

From the refinement60 directory:

```sh
python speed_study/steering.py --jobs 4 --out /tmp/steering-main.json
python speed_study/steering.py --jobs 4 --seeds 104 105 106 --speeds 0.30 0.45 0.50 0.55 --out /tmp/steering-holdout.json
python motion/run.py six-headings --turn-profile gentle --speed-mps 0.45 --seed 103 --model models/mid160/physics.xml --record --out /tmp/160-course
```

The evidence retains per-foot raw and rolling maxima, degrees of roll and pitch, loaded-contact slip, the fraction of time with each count of supporting feet, hold and endpoint results for all six requests, and actual speed and stop time. The additional seed panel uses the same course and controller. Add `--raw-dir PATH` to retain the full per-segment runner outputs besides the compact report.
