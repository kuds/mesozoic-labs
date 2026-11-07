# Round-one design review

**Status:** awaiting the owner's answers to the questions in §9. Nothing here is a final design, and
no controller training has started.

The illustrated version of this review is the review page, which is published separately and
linked in the delivery message. It uses the same numbers and plates. Every table below is
generated (`round1/TABLES.md`), and every image is a render of a compiled model in
`round1/review/mjcf/` (the plates are in `round1/review/plates/`).

## How round one was run

1. **Stage A sweep:** 132 candidates. Every combination of 4 heights × 3 widths × 3 body families ×
   3 postures, plus, per height, a belly-battery variant, three proportion variants (anatomical
   tail, compact, large skull) and two leg layouts (side-by-side hip roll, and a serial leg like PR
   #662). Each candidate was built with real component envelopes, then sized until it converged:
   loads, then drives and battery, then rebuild. It then got power, runtime and cost estimates.
2. **Review set:** 27 candidates spanning sizes, families, postures, widths, proportions and
   layouts. Each was rebuilt from its Stage A selection (the model's SHA-256 matches the sweep
   record), then screened for clearance and sensor coverage and drawn as plates.
3. **Fixes during the round:** each screen found problems, and the generator was corrected before
   these results were produced. The fixes covered:
   * the sizing method;
   * the sensor layout;
   * the trunk shapes;
   * five packaging defects.
   The failing versions are recorded in `SCREENING.md` §2 and the decision log (F3–F6).

## 1. Findings

1. **Size sets the drive class, and the drive class sets mass and cost.** Reference robots (balanced
   width, Substantial body, 10°):

   | Class | Crown / hip axis / length | Mass | Hip-pitch hold | Hip-pitch drive | Bus, battery | Walking power | Runtime | Cost nominal / high |
   |---|---|---|---|---|---|---|---|---|
   | 60 cm | 60 / 40 / 153 cm | 11.4 kg | 4.5 N·m | MyActuator X4-36-L (incumbent class) | 24 V, 6S 3.3 Ah | 96 W | 34 min | $15.6k / $21.6k |
   | 75 cm | 75 / 50 / 186 cm | 15.7 kg | 9.9 N·m | RobStride 03 | 24 V, 6S 3.3 Ah | 113 W | 29 min | $18.0k / $25.4k |
   | 100 cm | 100 / 66 / 247 cm | 33.6 kg | 27.0 N·m | MyActuator X12-320 | 24 V, 6S 5 Ah | 171 W | 29 min | $30.1k / $41.5k |
   | 120 cm | 120 / 80 / 297 cm | 46.1 kg | 51.2 N·m | MyActuator X12-320, past its planning limit (1.06) | 48 V, 2 × 6S 3.3 Ah | 209 W | 31 min | $38.0k / $52.7k |

   * The 60 and 75 cm classes share the bus, the battery and most drives. Only the 75 cm hip pitch steps
     up, which is where the earlier "9 N·m vs 6 N·m" gap appears and is resolved, with a larger drive.
   * The step from 75 to 100 cm more than doubles the mass (2.1×). The hip-pitch load grows 2.7× and moves the
     hips into a heavier drive class, while structure and coverings scale with size.
   * Per the owner's note (P8), the larger sizes stay in as full options. Their cost is reported, not
     used to rule them out.
   * Every high-side estimate is above $20k, including 60 cm at $20.4–23.3k.

2. **Hip pitch is the governing joint at every size.** At 120 cm it reaches the strongest single
   catalog drive. The Tapered body fits (0.93–0.98 of the limit). The Substantial and Hybrid bodies
   exceed it by 1–7 %. Round-two options, in `SCREENING.md` §3:
   * two drives per hip;
   * a reduction stage;
   * the Tapered body;
   * the anatomical tail;
   * bench data on the sustained-torque fraction.
3. **The tail is a counterweight.** The anatomical-length tail moves the centre of mass back from
   0.12–0.15 H to 0.09–0.11 H and cuts the hip-pitch hold by 24–27 %. That is 27.0 → 20.2 N·m at
   100 cm, which drops the hip drive class and saves $3.8k nominal. The cost is 0.3–0.6 m more length
   and turning sweep. The compact variant does the opposite: +26–40 % hip-pitch load.
4. **The hip layout decides single-support feasibility.** With both hip-roll motors on the centreline
   (tandem, the default), the 60 cm single-support hip roll is 2.9 N·m. The serial PR #662-like leg
   needs 12.8 N·m, the earlier gap reproduced. The side-by-side layout has no balanced single-support
   pose at 60–100 cm.
5. **The tail-base battery is worth keeping.** Moving the battery to the belly raises the hip-pitch
   hold by 18–78 % (4.5 → 8.0 N·m at 60 cm). It also pushes the centre of mass forward to 0.17–0.18 H,
   and the belly cassette has no straight removal path.
6. **Width is mostly a hip-roll and stability trade.**
   * Hip-roll hold grows with hip spacing: narrow / balanced / broad is 2.5 / 2.9 / 3.5 N·m at 60 cm and
     15.2 / 18.5 / 33.0 N·m at 120 cm.
   * At 120 cm the broad variant costs +8.9 kg and 11 min of runtime, and its reference walk is
     marginal (CoP −4 mm).
   * At 100 cm the narrow variant fails clearance: the pelvis plates meet the thighs.
   * Every variant stands about 3–7 × wider than tyrannosaurid trackways (0.24–0.49 H vs about 0.07 H).
7. **Body family changes the coverings, not the mechanics.** Tapered is the lightest and cheapest. It
   saves 0.6 kg / $0.8k at 60 cm and 2.5 kg / $2.6k at 120 cm against Substantial. Hybrid sits in
   between and shows the machine. All three pass clearance at 60 and 100 cm.
8. **Posture is cheap mechanically and changes height.** On the same body, 0° / 10° / 20° moves the
   crown by about ±10 % (53.7 / 60.1 / 66.2 cm at 60 cm). Hip-pitch load changes by under 5 %. Raising the
   chest also raises the head camera.
9. **Sensors moved.** The head camera sees a person's torso at 1.5–3 m at every size, with only about 3–4 % of
   its view blocked by the robot. The snout hides the floor ahead of the feet to 1.9–3.3 m, so near-field
   sensing relies on the two ToF sensors (45–71 % of the foot path covered).
10. **Packaging is tight at 60 cm.** The 60 cm reference passes every clearance gate but has near-contacts
    within 3 mm while walking, and knee flexion usable to 0.62 rad. 75–120 cm have room to spare.

## 2. Body families, posture and proportions

![Family × posture, 100 cm](round1/review/plates/family_posture_matrix_100.png)

| Family | What is covered | 60 cm mass / covers / cost | 120 cm mass / covers / cost |
|---|---|---|---|
| Tapered | Back and flanks; open ventral abdomen; exposed lower legs | 10.8 kg / 0.99 kg / $14.8k | 43.6 kg / 5.7 kg / $35.4k |
| Substantial | Full outline: chest, gastral belly, pubic boot, thighs, calves, tail root | 11.4 kg / 1.41 kg / $15.6k | 46.1 kg / 8.1 kg / $38.0k |
| Hybrid | Skull, neck, chest front, pelvis, thighs; exposed rib hoops and spine; segmented tail | 10.9 kg / 1.07 kg / $14.9k | 43.1 kg / 5.9 kg / $36.3k |

What each family keeps and avoids from the earlier forms:
* All three keep a deep chest, a waist behind the ribs, a pubic keel between the thighs and muscular
  thighs.
* They avoid the earlier rejected forms: no rectangular chassis, and no barrel torso with a hanging belly.
* The head still reads small against the trunk in side view. The large-skull variant (+15 % skull length)
  is shown for comparison and costs nothing mechanically (hip-pitch hold +0–5 %).

Plates:
* `sizes_common_scale.png`, `sizes_height_normalized.png`, `sizes_silhouettes.png`
* `families_60.png`, `postures_75.png`, `widths_100_front.png`, `widths_100_top.png`, `proportions_100.png`
* `silhouettes_family_posture.png`
* one four-view plate per review candidate: `<id>_four_views.png`
* cutaways: `60-M-S-10`, `100-M-S-10`, `120-M-S-10`, `100-M-H-10`, `100-M-T-10`, `60-M-S-10-serial`
* `layouts_60_cutaway.png`

## 3. Hardware compromises, by size

* **60 cm:** the incumbent drive class fits. Packaging is tight: the thigh fairing, ankle motors and pelvis
  sit within 3 mm while walking, and the tail pivot moves back 48 mm to fit the battery cassette. The
  head camera is an OAK-D Lite (no lidar).
* **75 cm:** the same drive families as 60 cm. The hip pitch steps up to a RobStride 03 and the ankles to
  the X4-36-L. It has more room inside and the same battery class.
* **100 cm:** larger drives everywhere (CubeMars AK60-39, RobStride 03, MyActuator X12-320 at the hip),
  a 6S 5 Ah pack on 24 V, OAK-D Pro W, and a lidar under the throat. Arrangement mass (structure,
  bearings, linkages, mounts) reaches 12.9 kg and coverings 5.0 kg.
* **120 cm:** a 48 V bus, the strongest catalog hip drive past its planning limit, and 18.6 kg of
  structure and arrangement mass. The 100 and 120 cm cutaways show the layout.

## 4. Compute

The owner asked (P9) whether onboard compute beyond the Jetson makes sense. The survey and the
per-size runtime effect are in `COMPUTE.md` and on the review page.

## 5. What this round does not show

* No controller has walked any of these models. The reference walks size the loads only.
* Carpet, turning loads, slips, impacts and thermal limits are not screened (`SCREENING.md` §7).
* All masses, powers, runtimes and costs are model estimates from sourced part data and stated
  rules. None is measured.

## 6. Questions for the owner (round one)

1. **Size** for round two: 60, 75, 100 or 120 cm, or two side by side.
2. **Body family:** Tapered, Substantial, Hybrid, or a mix (which parts of each).
3. **Posture:** level, 10° or 20° chest-up.
4. **Proportions to carry forward:** anatomical tail, larger skull, narrower or broader stance, compact legs.
5. **Compute direction:** see `COMPUTE.md`.

The answers go into `DECISION_LOG.md` as P-entries, and round two refines the chosen direction and
repeats every screen.
