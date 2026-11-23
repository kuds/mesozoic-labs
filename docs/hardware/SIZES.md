# Sizes explored

The sizes the work settled on were a T. rex 75 cm tall (about 15 kg) and a Triceratops 60 cm tall at the frill (about
12 kg). Smaller robots saved little mass and ran out of room; a T. rex of 1 m or more needed a heavier class of drive.
Every figure is a simulation result or a desk estimate; nothing has been built, bought or weighed. Costs:
[PARTS_AND_COSTS.md](PARTS_AND_COSTS.md). Trial results and lessons: [LESSONS_LEARNED.md](LESSONS_LEARNED.md). Status
and records: [README.md](README.md).

## How size is measured

- **Crown height** (T. rex): floor to the top of the head in the standing pose, chest raised 10°. Size classes name it.
- **Frill height** (Triceratops): floor to the top of the frill in a neutral stance.
- **Body length**: tip of the snout (T. rex) or horn (Triceratops) to the tip of the tail.
- **Mass**: model total of catalog part masses, allowances and reserves; none weighed. Detail: [definitions][defs].

## T. rex

The T. rex was studied from about 27 cm to 120 cm tall. When the work paused on 18 Nov 2026, a 75 cm body was the lead
size. Going below 60 cm saved little; going past 75 cm cost far more than it gave.

**Concepts before the study (24–28 Oct 2026).** Unmerged pull requests sketched a mini, a medium named "50 cm", a 60 cm
skeleton with twelve leg motors and a one-metre concept, as packaging and simulation studies with unweighed masses
([#635](https://github.com/kuds/mesozoic-labs/pull/635), [#648](https://github.com/kuds/mesozoic-labs/pull/648),
[#649](https://github.com/kuds/mesozoic-labs/pull/649), [#651](https://github.com/kuds/mesozoic-labs/pull/651)).
Findings on the same stack (PR #663) scaled a 60 cm platform to 75 and
90 cm, estimated to run 13.8 and 9.8 min against 18.6 min; only the 60 cm one finished the first walking screen on both
floors. These are the PR's own summaries; the study could not reproduce its walking results and did not build on this
stack.

**The 60 cm posture study (7 Nov 2026).** Nine bodies (tapered, deep-ribcage and open-rib shapes at 0°, 10° and 20°
chest-up) were simulated: all but one stood, but none could shift its weight onto one leg in an open-loop test. Leg
geometry, not the covering, set the sideways hip load; round one moved both hip-roll motors to the centreline to cut it
(figures in [LESSONS_LEARNED.md](LESSONS_LEARNED.md)).

**Round one: four heights (7 Nov 2026).** The model sized 132 candidates: crown heights of 60, 75, 100 and 120 cm;
narrow, balanced and broad widths; three body families (Tapered, lightly covered; Substantial, anatomically balanced;
Hybrid, selective covers over exposed structure); 0°, 10° and 20° chest-up; and shape and layout variants.

- Height: from 75 to 100 cm the mass grew 2.1 times and the hip-pitch load 2.7 times, moving the hips into a heavier
  drive class. At 120 cm the hip-pitch need exceeded the strongest catalog drive in 17 of 33 bodies.
- Family: Tapered cut the hip-pitch load by 9–16 % and was cheapest at every size. Substantial was dropped.
- Posture: 10° was kept; it reads as alert and moves loads, power and cost by about 1 %. Level became a variant.
- Width: wider hips load the hip roll; narrow 100 cm failed clearance; broad 120 cm cost 8.9 kg and 11 min runtime.

**Round two: 60 and 75 cm (8 Nov 2026).** Twenty candidates (Tapered and Hybrid; base, narrow, larger-skull, compact and
level) carried a lower-power Raspberry Pi 5 computer, so their runtimes rose. A new check, the animal-pace walk (speed
scaled to leg length, Froude number 0.05: about 0.45 m/s at 60 cm, 0.50 m/s at 75 cm), failed 12, including all ten
75 cm bodies, whose drives already ran at 0.96–0.98 of their limits. On 8 Nov three were picked: two narrow 60 cm bodies
and a 75 cm Hybrid with four upgraded drives and bigger feet. In all three the knee's speed was the limit.

**Size study (7–8 Nov 2026).** No nearby size did better. At 57 and 54 cm the body saved only 0.27–0.57 kg, because
about half its mass is catalog hardware that does not shrink; the hip-roll motor met the power board, and the 54 cm knee
ran too fast. At 75 cm four targeted drive changes passed every check for about $1.0k more (estimate), while 20 %
headroom on every drive snowballed to about 20 kg. At 80 cm, 0.7–0.8 kg heavier, the hip pitch ran too fast at animal
pace.

**Why 1–1.2 m was put off.** The long-term aim is a robot 0.9–1.2 m (3–4 ft) tall, and round one kept 100 and 120 cm as
full options. They were left undeveloped on 7 Nov and put off until much later on 10 Nov, while 60 and 75 cm were pushed
in simulation. The step up: one-leg hip-pitch holds of 27.0 and 51.2 N·m against 9.9 at 75 cm, a 48 V bus at 120 cm, and
every 100 and 120 cm candidate over $20k nominal in the 7 Nov cost model. In the study, no body above 75 cm was
simulated walking with a balance controller.

**Where it ended.** On 14 Nov 2026 the 75 cm body was put first: in simulation it was the most forgiving, walking even
without balance feedback where the 60 cm bodies fell. On 17 Nov the 60 cm walking styles were parked: the change up to
0.40 m/s asked the knee for 1.06–1.10 of its speed limit. Trial results: [LESSONS_LEARNED.md](LESSONS_LEARNED.md).

| Height | Body (date, 2026) | Mass | Runtime | Outcome (simulation or desk estimate) |
|---|---|---|---|---|
| 27 cm | Mini concept (Oct) | 0.75–0.83 kg | – | Stood and crouched; no walking |
| 55–58 cm | Medium concept, named "50 cm" (Oct) | 8.0–8.2 kg | – | Concept only |
| 60 cm | Skeleton concept (Oct) | 9.21 kg | – | Stood and crouched; mass flagged for review |
| 100 cm | One-metre concept (Oct) | 33.2 kg | – | Concept with a priced parts list |
| 60 cm | Posture study, nine bodies (7 Nov) | 9.87–10.08 kg | – | Eight stood; weight shift failed on all |
| 60 / 75 cm | Round-one references (7 Nov) | 11.4 / 15.7 kg | 34 / 29 min* | Taken to round two |
| 100 / 120 cm | Round-one references (7 Nov) | 33.6 / 46.1 kg | 29 / 31 min* | Kept on record, not developed |
| 54 / 57 cm | Size study (7–8 Nov) | 10.3–10.7 kg | 48–50 min | Failed clearance; knee too fast at 54 cm |
| 60 cm | Picks, narrow Tapered / Hybrid (8 Nov) | 10.81 / 10.91 kg | 47.6 / 47.4 min | Picked; Hybrid dropped 11 Nov |
| 75 cm | Pick, Hybrid with targeted drives (8 Nov) | 14.85 kg | 36.7 min | Picked; lead size at the pause |
| 75 cm | 20 % headroom on every drive (7–8 Nov) | 19.7–20.0 kg | 62–69 min | Failed: battery could not be removed |
| 80 cm | Size study (7–8 Nov) | 15.6–15.8 kg | 36 min | Failed the animal-pace walk |
| 75 cm | Pick with a Jetson computer (15 Nov) | 14.96 kg | 29.5–32.9 min | Same exterior; computer choice left open |

\* Round-one runtimes assume a 31 W computer; later rows use the 9.5 W Pi set or the Jetson's power modes. Detail:
[round-one review][r1], [round-two review and size study][r2].

## Triceratops

The planned build is 60 cm tall at the frill, 1.2 m long and about 12 kg, chosen on 18 Nov 2026 over a 75 cm, 1.5 m
design. The larger robot's estimate was well above the project's target of about $20k landed (parts at list price plus
freight, import duty on the drives, sales tax and contingency), and 60 cm uses the 75 cm T. rex's drive set.

The 75 cm concept ([#669](https://github.com/kuds/mesozoic-labs/pull/669), 17 Nov 2026) was 0.43 m wide and 15.0 kg with
a 1.84 kg reserve. The kickoff parts list (18 Nov, sized by calculation) compared 15 and 20 kg builds: the 20 kg one
covered a 0.30 m/s walk and a 0.60 m/s trot, while the 15 kg one's sideways drives would overheat in a sustained trot
(101 % of their limit). The kickoff recommended 20 kg; it was not adopted. The other options were scaled roughly from
that list, not simulated; speeds follow leg-length scaling. Because of that rough scaling the lean 75 cm option ($28.6k)
does not match the full 15 kg list ($32.8k landed, [PARTS_AND_COSTS.md](PARTS_AND_COSTS.md)).

| Option | Frill / length | Mass | Leg drives, bus | Walk | Run time | Landed estimate |
|---|---|---|---|---|---|---|
| 75 cm, 20 kg (kickoff list) | 0.75 / 1.50 m | 20 kg | 4 X4, 6 X6, 2 X8; 48 V | 0.30 m/s | 25–35 min | $37.3k |
| 75 cm, lean (15 kg body) | 0.75 / 1.50 m | 15 kg | 8 X4, 4 X6; 48 V | 0.30 m/s | 33–47 min | $28.6k |
| **60 cm (chosen)** | 0.60 / 1.20 m | 11.8–12.2 kg | 10 X4, 2 X6; 48 V | 0.25–0.27 m/s | 43–60 min | $23.9k–$24.7k |
| 60 cm, all-X4 | 0.60 / 1.20 m | 10.2–10.6 kg | 12 X4; 24 V | 0.20–0.27 m/s | 25–34 min | $22.3k–$23.1k |
| 50 cm | 0.50 / 1.00 m | 9.0–9.4 kg | 12 X4; 24 V | 0.17–0.23 m/s | 28–39 min | $20.6k–$21.9k |

X4, X6 and X8 are MyActuator drive sizes. Landed figures are as compared on 18 Nov 2026, with a certified wireless stop;
the 20 kg row is the nominal of the full kickoff list. Run times are per battery set.

Why 60 cm: the size comparison judged it the best balance of size and speed, and it shares the T. rex's drive types and
counts, so one bench, one set of spares and one firmware serve both. The all-X4 60 cm version saved $1.2k of hardware
but had no heat margin at the elbows; 50 cm saved $2.2k–$2.6k of hardware against the chosen size but walks slower and
is a smaller animal. With a radio stop (about $250, estimate) in place of the certified one, the landed estimate fell to
$20.1k–$23.0k, depending on frame cost and import duty. As modelled it weighs 12.69 kg with a 10 % reserve, 0.49 kg over
its range. Detail: [60 cm parts list][parts60].

## Other concepts

Notes from July to September 2026 sized seven other robot concepts; only the Compsognathus became a robot model, in
simulation.

| Concept (date) | Size | Mass | Where it stands |
|---|---|---|---|
| Compsognathus robot, revision B (Sep 2026) | 35 cm tall, 47 cm long | 1.59 kg | Model; stands in simulation |
| Compsognathus walker (Jul 2026) | seven powered joints | 2.0–2.2 kg (2.5 kg target) | Paper; revision B is current |
| Velociraptor walker (Jul 2026) | 10 motors | 14–16 kg | Paper; judged able to walk, not sprint, on catalog parts |
| Raptor runner, full size (Jul 2026) | 1:1 | 13.5 kg | Paper; needs custom actuators |
| Raptor runner, scaled to 0.72 (Jul 2026) | 9 motors | 6.5–7 kg | Paper; running unproven |
| Juvenile Psittacosaurus on Pupper V3 (Jul 2026) | 25 × 20 × 22 cm crouched | ≤3.5 kg (3 kg stock robot) | Paper |
| Dibothrosuchus-inspired quadruped (Jul 2026) | 1.1–1.3 m with tail | 10–12 kg | Paper; width simulated |

The Compsognathus revision B has twelve servos, six per leg (hip yaw, roll and pitch, knee, ankle pitch and roll), and a
fixed head and tail. Detail: [design record](../reviews/COMPSOGNATHUS_ROBOT_DESIGN.md).

The Dibothrosuchus-inspired quadruped has a 45–50 cm body and twelve CubeMars AK80-9 drives, about half its mass. A
simplified simulation study on 25 Jul 2026 compared shoulder and hip spacings of 180, 210 and 240 mm and chose 210 mm,
the narrowest that passed a reference sideways shove. Detail: [scaling and alternatives][rsa],
[walking parts estimates][bom], [chassis study script][chassis].

## What size taught

- Torque grows roughly with size to the fourth power (mass with the cube), and a biped needs about twice the per-joint
  torque of a quadruped of the same mass, because fewer legs carry the load.
- Catalog actuators are at their edge for a 5–7 kg running biped (hip peaks of 30–55 N·m) and run out at 13.5 kg
  (150–225 N·m); a 10–12 kg quadruped stays on catalog parts. For a walking T. rex they began to run out at 120 cm.
- Going down saves little: about half of a 54–60 cm T. rex is catalog hardware that does not shrink, and the
  Triceratops's drives, computer, safety and bench gear barely shrink (no demonstration-ready size under about $17.6k:
  parts plus the T. rex cost model's 7.2 % contingency, which covers freight, duty and tax).
- Speed grows only with the square root of size: at the same gait 75 cm is 11–12 % faster than 60 cm.
- Re-size the parts with the body: the larger versions in #663 appear to have kept the 60 cm drives and battery, so
  their short runtimes say little, and blanket headroom snowballed where targeted drive changes worked.

[defs]: https://github.com/kuds/mesozoic-labs/blob/87bafc864fefb16b040bd7a997fea8273bfc6649/hardware/trex_form_study/DEFINITIONS.md
[r1]: https://github.com/kuds/mesozoic-labs/blob/87bafc864fefb16b040bd7a997fea8273bfc6649/hardware/trex_form_study/ROUND1_REVIEW.md
[r2]: https://github.com/kuds/mesozoic-labs/blob/87bafc864fefb16b040bd7a997fea8273bfc6649/hardware/trex_form_study/ROUND2_REVIEW.md
[rsa]: https://github.com/kuds/mesozoic-labs/blob/87bafc864fefb16b040bd7a997fea8273bfc6649/docs/hardware/RAPTOR_SCALING_AND_ALTERNATIVES.md
[bom]: https://github.com/kuds/mesozoic-labs/blob/87bafc864fefb16b040bd7a997fea8273bfc6649/docs/hardware/HARDWARE_BOM.md
[chassis]: https://github.com/kuds/mesozoic-labs/blob/87bafc864fefb16b040bd7a997fea8273bfc6649/environments/shared/scripts/hw_chassis_study.py
[parts60]: https://github.com/kuds/mesozoic-labs/blob/7bfeef5334d0d7484e5428f1b77d2d1847d35c8c/hardware/triceratops_build/design/PARTS_60CM.md
