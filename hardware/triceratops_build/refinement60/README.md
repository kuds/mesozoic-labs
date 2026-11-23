# 60 cm open-rib Triceratops

This is the first foreleg mock-up candidate for the museum demonstrator. Frill crown height is 600 mm, shoulder and hip pivots are 280 and 320 mm above the floor in the neutral pose, and modeled mass is 12.844 kg. The forelegs use X4 roll, X4 pitch and X6 flexion through a 1:1 linkage. Each hind leg uses three X4 drives.

![Previous and revised forelegs at the same scale, with side and front views of the motor placement](media/placement.jpg)

[Four-beat walk, with four foot loads](media/walk.mp4) · [Moving turns, with requested and actual headings](media/steering.mp4) · [Printable design sheet](media/design-sheet.pdf) · [Foreleg bench plan](../bench/BENCH_PLAN_FORELEG.md)

## Motor, link and silhouette

Each X6 moves 20 mm aft, 5 mm outward and 6 mm upward in neutral upper-link coordinates. The bracket becomes a 3 mm plate. The upper and lower pivot spans remain 128 and 184 mm. Two 28 mm cranks stay coplanar, with 81.50 mm between rod pins at neutral. The forward rib, shoulder panels and edge of the dorsal covering are relieved around the drives. The sculpted forearm covers about 114 mm of the lower link and meets a tapered wrist cuff. The cuff adds no joint.

The robot's upper foreleg to upper hind leg ratio is 0.681. The corresponding humerus and femur measurements for Naturalis specimen RGM.1332500 give 0.682. Its 628 mm ulna scales to 114 mm using the robot’s 188 mm upper hind leg. Individual fossil bones and robot joint centres have different endpoints. The rounded back and short forelimb are also visible in the Minnesota mount. [Naturalis measurements, Table S1](https://repository.naturalis.nl/pub/800493/Jimmy-de-Rooij-2024-Bringing-fossils-back-to-life-A.pdf) · [Science Museum of Minnesota mount](https://assets.ctfassets.net/x2t7lek2vf7h/77vRqtPqnjM1sVlcFhKLRF/2d2bca550c9f2d379549e929cfaedc74/paleontology-publications-mounted-skeleton-of-triceratops-propsus.pdf)

Across 4,913 joint combinations per side the X6-to-upper-strut gap changes from a 9.20 mm overlap to 8.70 mm of clearance. The revised plate clears the rail by 4.68 mm. A separate calculation that moves both crank pins leaves 3.50 mm between the ideal rod and the motor. The housing clears the front sculpted rib by 4.97 mm and the back covering by 7.08 mm in the sampled exterior poses. The elbow is actuated directly in the dynamics model. The film displays a static rod graphic. Verify real rod ends, cable bends, guards, plate deflection and the stop chain under load before selecting production dimensions. [Geometry evidence](evidence/geometry/)

## Walking, balance and direction

Both four-beat walks use a 96 mm stride. The revised walk lowers commanded swing lift from 32 to 26 mm and shortens the interval between successive foot lifts from 1.34 to 1.07 s. It also reduces lateral body travel and removes the rearward offset on the first two setup swings. Tests that changed only the mount or only the gait attribute most of the reduction in sway to gait timing and targets.

| Measure | Previous | Revised |
|---|---:|---:|
| Passing slow walks, eight seeds on each of two contact settings | 16/16 | 16/16 |
| Nominal straight walking speed | 13.8 mm/s | 17.3 mm/s |
| Mean side-to-side body excursion | 145.2 mm | 127.9 mm |
| Greatest planted-foot slide | 9.16 mm | 7.36 mm |
| Minimum geometric support margin | 38.9 mm | 34.2 mm |
| Greatest slow-walk normal load on any foot | 91.4 N | 87.8 N |
| Static front weight share | 55.50% | 54.88% |

The geometric margin measures the horizontal centre of mass against the ankle triangle during a single swing. The separate heading course uses a 0.25 m/s diagonal gait with 10 ms of command delay. All six revised runs finished upright, and four acquired every heading within ±5° for one second. The other two missed the final return to straight. The faster course reached 355 N on one foot in the revised plant. [Matched gait and load results](evidence/gait-comparison.json)

Both designs acquired all six in-place pivots. A right pivot at seed 102 put **580 N on the right forefoot** of the revised plant versus **572 N** on the reference. This is the largest instantaneous modeled load across the tested routines. The largest 10 ms mean was 400 N versus 404 N. Include the [1 kHz revised pivot trace](evidence/pivot-right-102-refined.csv) in the guarded foreleg test. The simulations covered level ground. The shoulder opening needs physical guarding, and actual heading, traction and stopping require a caught test on the venue floor.

## Run the selected model

From this directory, use Python 3.14 and the pinned dependencies. The default walk in the refinement runner uses the selected gait.

```sh
python -m pip install -r requirements.txt
python verify.py /tmp/triceratops-refinement-check
python motion/run.py walk --model models/refined/physics.xml --seed 11 --record --out /tmp/refined-walk
python motion/run.py course --model models/refined/physics.xml --seed 101 --record --out /tmp/refined-course
```

Load [the refined dressed MJCF](models/refined/triceratops.xml) to inspect the shoulder opening or [the previous dressed MJCF](models/reference/triceratops.xml) for comparison. Keep both model directories together so their shared textures and meshes resolve. The GitHub model check verifies the dressed and physics variants, load balance at rest, two straight walks, both pivot directions and one moving course. Run the [full multi-seed comparison](motion/README.md) or [regenerate the geometry and clearance checks](geometry/README.md) for the broader evaluation.
