# How fast can the 60 cm Triceratops walk and turn?

The two robots have the same chassis, 184 mm lower forelegs, rocker feet, motor limits and 600 mm frill height. Only the foreassembly differs as described in the [128 versus 160 mm build comparison](../mid160/README.md). All speeds and loads on this page come from floating-base MuJoCo simulations.

[Rehearsed six-heading course at five cadences](media/rehearsed-speeds.mp4) · [Four-beat walk at increasing cadences](media/walk.mp4) · [Six headings with increasing diagonal-gait speeds](media/steering.mp4) · [Live four-beat steering experiment](media/fourbeat-turns.mp4) · [Rehearsed four-beat course](media/rehearsed-course.mp4) · [Speed and load plots](media/speed-limits.png)

## A natural walk and the museum operating choice

For this build, a convincing slow walk has distinct footfalls, weight on the three stance feet through each swing, smooth trunk motion and a small amount of planted-foot slide. The exhibit gait's existing criteria also require at least 10 mm of static support margin and no more than 10° of tilt. This is a robot performance test. Fossils do not establish a precise preferred speed for this miniature robot. The use of timing and real support contacts to describe a quadrupedal gait follows the observational approach of Hildebrand. [Gait study](https://pubmed.ncbi.nlm.nih.gov/5844074/) · [Existing exhibit-walk criteria](../motion/README.md)

**Use the 160 mm geometry for the first instrumented foreleg. Use 22 mm/s (1.3 m/min) as the straight-walk target when the assembled chassis is tested under a catch.** At 1.25× cadence both builds passed every walk criterion in 12/12 trials each. The 128 / 160 mm builds stayed within 1.04° of level, kept at least 33.8 / 34.1 mm of geometric support margin, and reached 89 / 94 N on their most heavily loaded foot. Front-elbow RMS torque reached 3.71 / 4.59 N·m. The longer link has an advantage in moving-turn loads and a higher thermal demand at the elbow. The shorter link has an advantage in straight walking at 1.40×, around 25 mm/s, passing 10/10 versus 9/10. Test elbow temperature, stopping and measured contact forces on the fixture before selecting an operating speed for visitors. [Straight-walk measurements and limits](EXHIBIT.md) · [Foreleg commissioning plan](../../bench/BENCH_PLAN_FORELEG.md)

At 1.50×, both models can produce a convincing 26–28 mm/s straight pass. Across seeds, the strict footfall sequence held in 9/10 runs at 128 mm and 7/10 at 160 mm. At 1.75× the films show rocking with conspicuous weight transfers. Peak loads reached 363 / 389 N and neither build passed. At 1.875× all twelve primary trials overturned. The fastest clean straight pass in a single film overstates repeatable performance. [Straight-walk film](media/walk.mp4) · [Primary trials](walk.json) · [Additional seeds](walk-holdout.json)

## Following the six heading commands

The 40 s course starts straight and commands +15°, +35°, +10°, −15°, −35° and 0°. Each must be held within ±5° of true chassis bearing for a continuous second inside its original time window. Sensors and motor commands retain their noise and 10 ms leg delay. The live controllers use the noisy gyro estimate for steering. Score every course through its final command or its first simulator stop. The native steering input range ends at 0.30 m/s. The tests above that range expand only the simulation input encoding.

| Command | 128 / 160 mm measured travel speed | 128 / 160 mm maximum single-foot load | What happens |
|---|---|---|---|
| 0.25 m/s | 0.246 / 0.245 m/s | 416 / 281 N | All six headings on both builds |
| 0.30 m/s | 0.295 / 0.294 m/s | 251 / 201 N | Top of stock input range, all headings |
| 0.45 m/s | 0.449 / 0.447 m/s | 445 / 321 N | Both hold all headings in 12/12 runs each |
| 0.50 m/s | 0.470 / 0.458 m/s | 782 / 601 N | Both 12/12, with rising slip and flight |
| 0.55 m/s | 0.472 / 0.466 m/s | 771 / 639 N | 128 mm passes 12/12, 160 mm passes 7/12 |
| 0.60 m/s | No full runs | 1,007 / 949 N | All twelve primary runs hit the tilt stop |

Speed and force columns use the six primary trials per build. Heading counts include additional seeds where stated. Raising the command from 0.25 to 0.35 m/s lowered the highest instantaneous foot load for both builds, while their worst loaded-foot slip speeds increased. Over six trials per build at 0.45 m/s, worst 95th-percentile slip was 0.24 / 0.25 m/s. At 0.50 m/s it rose to 0.343 / 0.335 m/s, with brief intervals in flight. A successful heading hold does not require clean traction or an animal-like gait. The films show the contact timing and rebound. Forces are sole-normal loads sampled at 1 kHz. They exclude body impacts and a fall's landing after the steering simulator stops on tilt. [Per-trial results, 10 ms loads and added seeds](STEERING.md)

## Can a four-beat walk also steer?

The live staggered schedule uses the same gyro steering adapter, a 1.75 s stride, 80% planned stance, 18 mm swing lift and 30 mm of lateral trunk transfer. The swing order runs hind then fore on one side, followed by hind then fore on the other. Each 40 s trial uses the ordinary six-heading targets. We measured the contact response on the robot instead of inferring the gait from the schedule. Each complete swing is checked at the 50 Hz recorded poses using the exhibit rule. The intended swing foot must unload for at least 80% of samples, and all three other feet must stay loaded. A loaded foot carries at least 5% of the modeled weight. Signed distance from the actual centre of mass to the three intended stance ankles gives the geometric margin.

Across seven commands from 0.025 to 0.24 m/s per geometry, seeds 103 and 105 on nominal and randomized conditions supplied four courses per speed. At a 0.08 m/s command both builds hit all six headings on all four runs and measured about 0.073 / 0.072 m/s after setup. In the filmed run, three or more feet were loaded for 66% / 67% of this interval. Only 18 of 84 complete swings on each build met the ordered-contact criterion. At least one stance triangle had negative margin in every run with a complete swing. No tested live staggered pace consistently met the exhibit contact and margin criteria. A 0.16 m/s run on the 128 mm build missed two headings and reached 1,239 N on the left forefoot, 684 N over 10 ms. [Every staggered run](four-beat-results.json) · [Overload trace](fault-staggered-128.csv)

### Rehearsed six-heading walk

For a museum route with known bearings, we bent the native four-beat trajectory around the same six commands, placed each swinging foot for the upcoming turn and shifted the trunk over the three intended stance feet. The chassis moves freely in contact with the floor. The joint targets use the known course and clock. An unexpected change of direction would require a new plan. Each figure below covers matched seeds and contact settings, with speed and force measured from the first changed command at 3 s through 40 s.

| Cadence | Mean achieved mm/s, 128 / 160 | Runs keeping ordered contacts, 128 / 160 | Largest single-foot load, 128 / 160 |
|---|---:|---:|---:|
| 1.00× | 17.4 / 17.5 | 0/6 · 0/6 | 224 / 200 N |
| 1.25× | 21.3 / 21.4 | 0/6 · 0/6 | 145 / 148 N |
| 1.40× | 25.3 / 25.2 | 3/6 · 3/6 | 137 / 139 N |
| 1.45× | 27.1 / 27.1 | 5/12 · 8/12 | 126 / 128 N |
| 1.50× | 28.4 / 28.3 | **11/12 · 7/12** | **138 / 138 N** |
| 1.65× | 30.3 / 30.3 | 0/6 · 0/6 | 129 / 130 N |

**All 96 simulations stayed upright and held all six bearings.** The contact column counts complete runs in which the intended swing foot unloaded and the other three feet supported the chassis through every swing. A run also had to keep at least 10 mm of geometric margin. At 1.50×, both builds stayed within 1.8° of level with at least 27.6 mm of margin. The worst 10 ms loads were 130 / 131 N. Independent 1 kHz traction replays at 1.50× found at most **8.48 / 9.02 mm** of net slide at the planted contacts, inside the 10 mm limit. Neither model saturated an actuator at a judge sample. The combined score including headings, upright posture, margin, contacts, sliding and actuator effort therefore stayed **11/12 versus 7/12**. Every miss at this setting came on the stiffer floor. [Per-trial traction and saturation](choreography-traction.json) The inverse kinematics missed its ankle target at 14 of 7,396 reference samples, by up to 7.8 mm. At 1.65× both builds still made the turns, but reached almost 9° of tilt and lost their intended supports several times. At 1.25× three feet bore weight about 99% of the time, yet every run lost the intended contact pattern on one or two swings. Peak foot force alone is a poor way to identify those missed steps. [Watch the side-by-side speed sweep](media/rehearsed-speeds.mp4) · [Full 1.25× course](media/rehearsed-course.mp4)

[1.00× and 1.45× measurements](choreography-speed.json) · [1.25× measurements](choreography.json) · [1.40×, 1.50× and 1.65× measurements](choreography-edge.json) · [1.45× and 1.50× additional seeds](choreography-holdout.json)

At 1.40× and 1.45× the traction replays likewise left the ordered-contact pass counts unchanged. All 36 runs met the slide and saturation criteria. One 128 mm run at 1.45× came within 0.001 mm of the modeled 10 mm slide threshold. [Traction at 1.40× and 1.45×](choreography-traction-edge.json) · [1.45× additional seeds](choreography-traction-holdout.json)

## Reproduce

Run the following commands from the refinement60 directory with its pinned physics dependencies. The optional plot requires Matplotlib. The movie renderer requires Pillow, an offscreen MuJoCo context and H.264 FFmpeg. The steering evidence includes the full portion observed on each early stop. Mean achieved speed in the diagonal-gait table uses completed trials. Achieved exhibit speed uses forward displacement over trial time excluding the two standing holds. Actual speed for the staggered course uses projected progress after the straight setup.

```sh
python speed_study/walk.py --records /tmp/trike-speed/walk --out /tmp/walk.json
python speed_study/steering.py --raw-dir /tmp/trike-speed/steering --out /tmp/steering-main.json
python speed_study/four_beat.py --speeds .025 .05 .08 .12 .16 .20 .24 --seeds 103 105 --records /tmp/trike-speed/fourbeat --out /tmp/four-beat-results.json
python speed_study/choreography.py --cadences 1 1.25 1.4 1.45 1.5 1.65 --records /tmp/trike-speed/rehearsal --out /tmp/choreography-main.json
python speed_study/choreography.py --cadences 1.45 1.5 --seeds 21 22 23 --records /tmp/trike-speed/rehearsal --out /tmp/choreography-holdout.json
python speed_study/choreography.py --traction --cadences 1.5 --seeds 11 12 13 21 22 23 --records /tmp/trike-speed/traction --out /tmp/choreography-traction.json
python speed_study/plot.py
python speed_study/render_movies.py rehearsed-speeds --records /tmp/trike-speed/film-inputs --record-missing --out /tmp/rehearsed-speeds.mp4
```

The films play at real time, with common scale and matched seeds. Ribbons show measured foot contacts. The speed sweeps show matched excerpts at the labeled cadences or commands. The live four-beat steering film shows the full 0.08 m/s command trial at nominal seed 103. [Film settings](media/stages.json)
