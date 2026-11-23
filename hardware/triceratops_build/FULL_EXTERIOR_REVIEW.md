# Reviewing the fully covered 60 cm Triceratops

Option C is 60 cm to the frill, with a 1.2 m overall length target, ten X4 drives, two X6 drives and the D19 capsule rocker. The covered model is a proposed dinosaur exterior for this exact plant. The current procurement decision is one instrumented foreleg. The numbers below are simulation results or budget estimates.

## What gives the robot its proportions?

The shoulder and hip pivots sit at the same 320 mm height, 520 mm apart. All four legs have 188 mm upper and 184 mm lower links. The mirrored bends and equal-sized ground contacts are visible even with the covers removed. A suit can hide an axle, adjust the beak line and taper the belly into the tail. Moving an elbow or bringing the feet closer together requires different kinematics.

As a specimen-specific comparison, the St Paul *T. prorsus* mount has a 0.74 m humerus and a 1.14 m femur, a fore/hind upper-bone ratio of 0.65. The robot’s corresponding link ratio is 1.00. The specimen’s socket spacing is 2.46 m over a 7.92 m curved beak-to-tail measurement. The robot spends about 43% of its nominal overall length between the pivots, compared with about 31% on the mount. The length conventions and the anatomical and mechanical endpoints differ. The specimen also includes restored elements. These comparisons explain a design tension without prescribing a new wheelbase. [Mounted skeleton, principal measurements](https://assets.ctfassets.net/x2t7lek2vf7h/77vRqtPqnjM1sVlcFhKLRF/2d2bca550c9f2d379549e929cfaedc74/paleontology-publications-mounted-skeleton-of-triceratops-propsus.pdf) · [Museums Victoria articulated T. horridus](https://collections.museumsvictoria.com.au/specimens/2491830)

In the initial and revised neutral exterior the same pivot spacing gives a direct comparison. Beak to midline frill crown increased from 337 to 423 mm while torso width fell from about 300 to 254 mm. The covered head-length/pivot-spacing ratio improved from 0.65 to 0.81; the St Paul specimen is about 0.90 (2.21/2.46). The adjusted beak-to-tail side projection is 1,186 mm. The shoulder and hip centres still span 44% of that distance. The specimen comparison is an approximation because the fossil length follows the curvature and the robot's side measurement is straight.

A second, associated *T. horridus* specimen (Naturalis RGM.1332500) has a 708 mm humerus and 1,038 mm femur, a ratio of 0.68. Both fossils support the observation that equal robot upper links influence the crouched side profile. Bone length and mechanical joint spacing are different measurement conventions. [De Rooij, Table S1](https://dspace.library.uu.nl/bitstream/handle/1874/437182/phd_thesis_jimmy_de_rooij_naturalis-utrecht_university_2024_uses_307_-_65e85ade07002.pdf?isAllowed=y&sequence=1)

Across four walking poses and seven camera angles, the exterior covered the rails, crossmembers, drives and wiring in all sampled views apart from three image pixels on the right front drive crank. The forward range-sensor carriers also remained visible. Final bracket, seal and moving-panel clearances need a physical mock-up.

Use the covered side view and landmarks to judge the rounded caps, barrel-shaped torso, raked frill and jaw before making tools. Keep the current hard points for this foreleg bench. A shorter trunk, differentiated fore/hind links or altered ride height belongs in a separate kinematic study with actuator envelopes, neutral balance, contact timing and walking stability rerun. [Exterior source and checks](renders/full_exterior/README.md)

## Mechanics and foot loading

The closed plant weighs 12.694 kg, including its allocated shells and reserve. The centre of mass sits about 28 mm forward of mid-body and 286 mm above the floor at settled stand. The forefeet take 55.3% and the left side 49.8%. For the nominal level trot, seeds and time windows matter. These forces are summed normal reactions by foot on level ground. They do not resolve pressure within the real rubber rocker.

| Simulated load, N | Left fore | Right fore | Left hind | Right hind |
|---|---:|---:|---:|---:|
| Settled stand | 34.28 | 34.57 | 27.69 | 27.98 |
| Level trot, time mean | 33.14 | 34.43 | 28.65 | 28.31 |
| Level trot, 99th percentile | 175.5 | 208.9 | 93.5 | 92.6 |
| Level trot, single-sample peak | 275.2 | 316.5 | 152.5 | 203.8 |

The frozen walk60 controller was commanded to 0.25 m/s with 10 ms delay, nominal model settings and observation noise. Forces are sampled at 1 kHz over seconds 2–20 for seed 101. The left-fore/right-hind pair carried 49.34% of the total impulse. Four scales under a level, painted assembly can check the static shares. A synchronized force plate, angle record and high-speed view under the suspect forefoot can separate rolling, recontacts, slip and toe or cover drag. Weigh the dressed robot in neutral, with its head turned, and with one leg unloading under restraint. [Study data](review60/README.md) · [Sole design and acceptance](sim/REPORT_FEET60.md)

The full capsule proxy intersects the lower strut by 21.6 mm. The real sole needs a retained half-capsule and clearance through ankle pitch. The model has only 1 mm between battery packs and 0.5 mm between the NVMe tray and stop-relay block. Inspect the elbow axle/crank connection, rod-end misalignment and play, camera sightline past the neck cradle, moving tail-servo envelope, cable loops and room to extract the battery. The modeled elbow joint cap of 37.8 N·m differs from bearing loads. In the demo course, the peak front-foot and front-elbow reactions reached about 406 and 416 N, with front-pitch bending of 39.1 N·m. [Packing](sim/REPORT_60CM.md) · [Demo loads and thermal assumptions](rl/README.md) · [X4/X4/X6 test plan](bench/BENCH_PLAN_FORELEG.md)

## What the walking and face communicate

In the model-based clip the robot moves forward cleanly but reads as a brisk mechanical trot. The feet travel in alternating diagonal pairs, the pelvis travels nearly level, and the frill faces straight ahead. The body-height 1st–99th percentile range is just 4.8 mm. The front feet briefly unload and reload about three times per scheduled stride (4.22 and 4.28 contact onsets/s vs 1.39/s per hind foot, using a 10 ms debounce). A sustained step and a short rebound produce different impressions. A second filter merging interruptions shorter than 50 ms gives diagonal-only support about 73% of the time. The right forefoot’s peak rose from 317 to 386 N when the command was reduced to 0.15 m/s. Slowing this controller did not produce the visual weight of a deliberate walk. [Selected source clip](https://github.com/kuds/mesozoic-labs/blob/408d5c59/walk60_capsule_trot_0.25_1x.mp4) · [Replay data](review60/README.md)

The registered four-beat crawl offers a different motion. Eight selected-foot trials advanced at 0.0182–0.0184 m/s, with a minimum mid-swing support margin of 49.9–51.5 mm and maximum body tilt below 0.7°. Use it for a deliberate approach or grazing stop. For an intermediate exhibit walk, tune swing deceleration and the foot’s real contact compliance, then compare load peaks, distinct contacts, slip, speed and undistorted side/front video. Keep grounded feet fixed when adding apparent weight transfer. [Crawl and foot trials](sim/REPORT_FEET60.md)

The exploratory selected-foot terrain study ran eight 20-second seeds per point. Mean speed fell from 0.249 m/s on flat ground to 0.214, 0.203 and 0.167 over 1, 2 and 3 cm steps and to 0.135, 0.117 and 0.090 on 3°, 5° and 8° ramps. All 56 completed upright. Right hind-knee RMS torque averaged 1.12 times the modeled continuous rating at both 3° and 5°. Include these profiles in the hind-X4 thermal test. Flat floors remain the supported engineering target for the current walking design. [Terrain conditions and results](review60/README.md)

Default accessory commands are neutral. The visible thump travels into the head, but the eyes do not choose a target. The covered preview adds a pivoting jaw and two movable eyelids as visual mechanisms. For a readable exhibit cue, lead with the head, hold gaze briefly, blink once, then let the tail settle. At a feeding stop, lower the beak before opening it and let the closing movement take longer. The original build prices seven expression axes, including neck and tail pairs, beak and two lids. Eyeball rotation would require additional hardware.

A separate exploratory test commanded 12° head yaw, 5° nod, 4° opposing tail yaw and 2° tail pitch. All 12 runs across three seeds, neutral/cued and nominal/+400 g, stayed upright for 12 s. At nominal mass the maximum tail-pitch RMS rose from 0.216 to 0.293 N·m with the cue. This test covers neck/tail dynamics. The eyelid and beak skin still need linkage, stall-current and clearance checks. [Cue and mass experiment](review60/README.md)

## What complete coverage changes

| Team-finished shell scenario | Estimated shell line | Change vs current | Approximate entire robot, landed |
|---|---:|---:|---:|
| Low | $840 | −$530 | $19,470–$22,353 |
| Existing budget | $1,370 | $0 | $20,105–$22,988 |
| High | $2,330 | +$960 | $21,255–$24,138 |

The budget already includes nylon body and limb panels, head, frill, tail, soft interfaces and team-applied paint materials. These scenarios use the kickoff low/nominal/high shell-plus-soft estimates, the project’s 0.64 area factor, nearest-$10 rounding and $150 finish materials. The landed change uses the worksheet’s 1.1984 factor for non-drive cost. They exclude team labor. A previous larger-build artist allowance of $600–$2,500 corresponds to an illustrative additional $450–$2,350 before overhead if a vendor replaces the $150 materials line; whether materials are included needs a quote. Price shell splits, inserts, gaiters, ductwork, surface prep, finishing, reprints and labor separately before fabrication. The model already carries $976 for expression and $90 for blowers/ducts. [Option C costs](budget/size_budget.py) · [Original shell and finish allowances](../triceratops_kickoff/PARTS_PREMIUM.md) · [Mass by assembly](design/PARTS_60CM.md)

The closed ledger allocates 0.967 kg for body/limb shells, ribs and seams. The head and tail have additional separate mass entries. At 1.07 g/cm³, adding 1 mm of silicone over just 0.10 m² adds 107 g before adhesive or paint. Put compliant material at flex folds and hard-panel overlaps. Taper lightweight hollow volumes and weigh the painted parts with inserts. Record each part’s position, especially at the ankles and ahead of the neck. [Material density](https://www.smooth-on.com/products/dragon-skin-10-medium/)

The revised model has about 2.23 m² of closed mesh surface, including concealed overlaps, both frill faces and closed ends. This figure cannot determine cloth or coating consumption without a panel layout.

Use removable belly and side panels, fasteners reachable through the service sequence, protected optical apertures, overlapping gaiters and baffled intake/exhaust paths. Keep the floor clear of skin except at the intended soles and belly skid during a supported stop. Mark the stop release and preserve access with the suit installed. Measure drive/compute temperatures, fan power, real airflow, motion current, skid function and battery removal time with the actual cover. The open-air project estimate was 106 minutes per pack at 66 W, about 92 minutes with 40 W hotel load. Neither is an enclosed-runtime rating. Check sound at the microphone with the fans running, and repeat the same profile covered and uncovered at 25 °C and 35 °C. Stage a quarter-body/one-leg sample before committing the full finish. [Demo power model](rl/README.md) · [Commissioning plan](bench/BENCH_PLAN_FORELEG.md)
