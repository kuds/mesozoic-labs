# Triceratops quadruped design study

A 75 cm Triceratops is the first build proposed here. Four load-bearing limbs put the shoulder pair beneath the frill. A 60 cm variant uses scaled structural geometry and keeps the purchased drives, battery and compute envelopes at their installed sizes.

| Neutral stance or goal | 75 cm prototype | 60 cm companion |
| --- | ---: | ---: |
| Frill crown height | 750 mm | 600 mm |
| Overall length | 1,500 mm | 1,200 mm |
| Overall width, rounded | 430 mm | 348 mm |
| Front-to-rear foot spacing | 650 mm | 520 mm |
| Foot track, centre to centre | 350 mm | 280 mm |
| Walking speed target | 0.30 m/s | 0.218 m/s |

## Design files

[Design package (PDF)](Triceratops-design-package.pdf) includes the mechanical and electrical proposal, leg-load calculations, test plan, dimensioned side and plan views (page 6), and a review of the T-Rex evidence (pages 7–9).

[Editable OpenSCAD envelope](Triceratops-envelope.scad) has assembled, skeleton, packaging and shell views. Select the crown height with `height_mm`. To render both neutral poses locally from this directory, run:

```sh
openscad -D 'height_mm=750' -o /tmp/triceratops-750.stl Triceratops-envelope.scad
openscad -D 'height_mm=600' -o /tmp/triceratops-600.stl Triceratops-envelope.scad
```

## MuJoCo models

[Open the two MuJoCo scenes and their control guide](models/README.md). The adult anatomical reconstruction and the 75 cm robot each have a free base, working joints and a four-foot stance check. The robot scene includes the motor housings, belt transmissions, control electronics and sensors.

## Evidence and open questions

The 75 cm T-Rex passed the full 0.30 m/s walking criteria in simulation at zero added delay. At 4 ms one of two controller fits passed. No tested speed passed at 10 ms. The corresponding Triceratops speed is a design target. [T-Rex Stage 2 results](https://github.com/kuds/mesozoic-labs/blob/42e6f5075193994edbd2d2441ae8933a5efde09a/hardware/trex_form_study/STAGE2_75_REPORT.md)

The Triceratops study covers stationary packaging, planar inverse kinematics and free-base stance in MuJoCo. Free-base walking, clearances throughout a stride, and physical performance remain unverified. The proposed 15 kg chassis puts 55% of its standing load on the front axle. The foreleg calculation calls for about 24.2 N·m briefly at the flexion motor. Verify the X6 drive and 1:1 belt under load on the intended 6S supply, including heating and command delay with both CAN buses carrying twelve drives’ traffic. [T-Rex bench protocol](https://github.com/kuds/mesozoic-labs/blob/42e6f5075193994edbd2d2441ae8933a5efde09a/hardware/trex_form_study/stage2/jetson75/BENCH_SHEET.md)
