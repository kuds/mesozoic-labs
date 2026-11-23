# Straight four-beat speed limits

The same 96 mm target stride, 26 mm swing height, joint targets and supports were used at each cadence. Time between planned foot lifts changes from 1.073 seconds at 1× to 0.859 seconds at 1.25×. The simulation still runs at 1 kHz. The independent once-per-pass neck gesture uses its original time base. Each seed changes initial joint offsets and sole friction. Each seed is repeated on the nominal and stiffer contact settings.

**Use 1.25× as the shared exhibit starting point.** It passed all 12 trials per geometry, reaching 21.7–22.4 mm/s at 128 mm and 21.8–22.5 mm/s at 160 mm. The geometric support margins were at least 33.8 and 34.1 mm. Worst front elbow RMS torque was 3.71 versus 4.59 N·m, and both builds remained within 1.04° of level.

| Cadence | Passed, 128 / 160 | Highest foot normal force, 128 / 160 | Largest 10 ms mean, 128 / 160 |
|---|---|---|---|
| 1.00× | 6/6 · 6/6 | 87.7 · 89.8 N | 84.7 · 86.2 N |
| 1.25× | 12/12 · 12/12 | 89.2 · 93.5 N | 86.3 · 90.3 N |
| 1.40× | 10/10 · 9/10 | 91.7 · 96.4 N | 88.6 · 93.0 N |
| 1.45× | 8/10 · 7/10 | 92.2 · 96.6 N | 89.1 · 93.3 N |
| 1.50× | 9/10 · 7/10 | 90.9 · 95.5 N | 87.9 · 92.3 N |
| 1.525× | 0/6 · 0/6 | 99.4 · 114.2 N | 95.5 · 109.7 N |
| 1.75× | 0/6 · 0/6 | 363.3 · 388.8 N | 327.8 · 349.8 N |

The primary panel uses seeds 11–13 on both floors at 15 settings spanning 1–2.5× (180 runs). The additional 36 trials use seeds 21–23 at 1.25×, and seeds 21 and 22 at 1.40×, 1.45× and 1.50×, on both floors. In the primary panel the fastest 6/6 setting for 128 mm was 1.50× (26.2–27.5 mm/s), and for 160 mm it was 1.40× (24.5–25.1 mm/s). Each missed one of its four additional runs at those respective speeds. Onsets vary with seed, contact and footfall timing. At 1.475× the 128 mm primary result was 5/6. Exact scores and individual feet are available in the [primary trials](walk.json) and [additional trials](walk-holdout.json).

The first missed rule is normally C4, even when the chassis still looks level. In each swing the commanded foot must carry less than 5% of body weight for at least 80% of that swing, while each of the three other feet stays at or above 5% at every 100 Hz judge sample. At 1.525× all 12 runs missed this test. At 1.75× the worst tilt reached 14.4° and 14.9°. At 1.875× both robots overturned in all six trials each, reaching more than 130° of tilt. Speed after overturning no longer represents walking performance.

At the deliberate 2.5× fault setting, the 128 mm model reached **780.5 N on the left forefoot** at 3.093 s on the stiff floor, seed 11, with a 659.3 N rolling mean over ten physics samples. This one run rocked to 33.6° and failed the gait criteria. The greatest foot force across the 160 mm fault panel was 404.0 N. The measured channels cover the feet; torso or shell impacts during a fall require independent fixture assessment. [1 kHz samples around the 128 mm fault](walk-fault-128-stiff11.csv)

From the refinement60 directory:

```sh
python speed_study/walk.py --jobs 4 --records /tmp/trike-walk-main
python speed_study/walk.py --holdout --jobs 4 --records /tmp/trike-walk-holdout
python motion/run.py walk --model models/mid160/physics.xml --walk-cadence 1.25 --record --out /tmp/trike-160-walk
```
