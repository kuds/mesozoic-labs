# Round-two design review

**Status:** awaiting the owner's answers to §8. Nothing here is a final design, and no controller
training has started (P7). The proposed next step, a bounded closed-loop probe, also waits for the
owner's go-ahead.

The illustrated version of this review is the round-two review page, published separately and linked in
the delivery message (`python -m designlab.review2_page` builds it). It uses the same numbers and plates.
The tables below are compiled from the generated tables in `round2/TABLES.md` and the scorecard in
`round2/SHORTLIST.md`, rounded for display. Every plate is a render of a compiled model; the plates are in
`round2/review/plates/`.

## How round two was run

1. **Candidates (P10–P15):** 60 and 75 cm crown height × Tapered and Hybrid bodies. Each base body
   (balanced width, 10° chest-up) has four single-change variants: narrow stance, larger skull, compact
   proportions (shorter legs and tail, larger skull) and the level 0° posture. The level variant is the
   same body pitched level, so its crown is lower (54 / 67 cm). That makes 20 candidates, all with the
   Raspberry Pi 5 + AI HAT+ compute (P14).
2. **Generator revision 2:** a camera placed by a search inside the skull (F8); packaging rules for the
   hips, tail motor, ankle drives and calf (F9); and twelve shape changes from an independent visual
   critique (F10). Round-one models still rebuild byte-identical.
3. **Sizing:** the round-one loop (loads → drives and battery → rebuild until stable). Its reference walks
   now converge: the centre of mass is held to 1e-6 m and the stride boundary is unwrapped exactly (F14).
4. **Packaging search:** a body that fails a walking clearance gate is changed and re-sized: the legs move
   out, or the ankle drives move down the shank. The narrow stance must also turn without contact (F16).
5. **Screens on every candidate:** clearance with the round-two gates, sensors, turning (F11), carpet (F13)
   and a behavior envelope (F12). Each rebuilt model's SHA-256 matches its sized record (20 of 20).
6. **Scorecard and picks:** gates, then 0–1 margin scores per behavior weighted by the owner's priorities, a
   stress test and a diversity rule (`designlab/shortlist.py`). An independent panel then cross-checked the
   picks (§6).

## 1. Results at a glance

All 20 candidates pass every round-two gate: clearance at home and in both walks, every drive within its
planning limits in the sizing walks and the target turns, a person's torso in view, at least 20 min of
runtime, the reference walk's centre of pressure on the pad and the carpet gate.

| Base body | Crown / hip axis / length | Mass | Walking power | Runtime | Cost nominal / high | Governing drive ratio |
|---|---|---|---|---|---|---|
| `60-M-T-10` Tapered | 60.1 / 40.5 / 151.0 cm | 10.85 kg | 71 W | 46.4 min | $14.8k / $20.6k | thigh twist, sustained 0.77 |
| `60-M-H-10` Hybrid | 60.1 / 40.1 / 144.9 cm | 10.96 kg | 71 W | 46.2 min | $14.9k / $20.7k | knee speed 0.79 |
| `75-M-T-10` Tapered | 75.1 / 50.6 / 183.7 cm | 14.90 kg | 86 W | 38.2 min | $17.4k / $24.5k | neck servo, sustained 0.97 |
| `75-M-H-10` Hybrid | 75.1 / 50.2 / 175.9 cm | 15.03 kg | 87 W | 38.0 min | $17.5k / $24.6k | thigh twist, sustained 0.97 |

Every candidate runs on 24 V with a 6S 3.3 Ah pack. Against the same four round-one ids, walking power falls
by 22–26 W, mostly from the Pi 5 compute (9.5 W budget against the Jetson's 31 W), so runtime rises from
30–34 min to 38–46 min. Mass changes by +0.09 to +0.20 kg and cost by less than $70.

**The 60 cm bodies keep about 20 % drive headroom; the 75 cm bodies do not.** At 60 cm the governing ratio is
0.77–0.79 on the base, narrow and level bodies, and 0.85–0.96 on the larger-skull and compact variants. At 75 cm
it is 0.96–0.98 on every body. On seven of them that is the thigh twist (CubeMars AK45-36) at its sustained
limit. On three it is the neck servo (ROBOTIS XM540-W150, 0.96–0.97 wherever it is selected); the larger-skull
and compact bodies need the stronger XM540-W270, which then runs at 0.76–0.80. Most of the scorecard's gap
between the sizes follows from this (§6).

Plates: `families_r2.png`, `bases_common_scale.png`, `bases_silhouettes.png`, `families_r2_cutaway.png`,
`variants_60_side.png`, `variants_75_side.png`, `variants_60_front.png`, `variants_75_front.png`,
`silhouettes_r2.png`, `r1_vs_r2_60_cutaway.png`, one `<id>_four_views.png` per candidate, and cutaways of
the four bases.

## 2. Behaviors (Stage 0 screens)

These are necessary-condition checks on prescribed motions and simple balance models. Failing one shows
that no controller can produce the behavior with that body and those drives; passing one does not show
that a controller can (`SCREENING.md` §8).

| Behavior | 60 cm (10 bodies) | 75 cm (10 bodies) |
|---|---|---|
| Target turns (0.35 rad/s walking, 0.5 rad/s in place), drives / IK / contact | all pass; governing ratio 0.77–0.96 | all pass; 0.96–0.98 |
| Thigh-twist speed for the turn in place (with the 1.15 margin, of 4.19 rad/s usable) | 2.70–2.81 rad/s | 2.87–2.97 rad/s |
| Friction needed in a walking turn, single support (yaw moment included) | μ 0.53–0.65 | μ 0.37–0.43 |
| Carpet gate (friction below 0.54, swing foot above the pile top) | all pass; need 0.24–0.28, toe ≥ 5.3 mm above high pile | all pass; need 0.22–0.24, toe ≥ 10.1 mm |
| Carpet swing target (0.06 H or 15 mm above the pile, not gated) | none; needs a 0.19 H lift (reference: 0.10–0.11 H) | none; needs 0.17–0.18 H |
| Servo-held stance with no balance control: low / medium / high pile | stands / tips / tips | stands / 4 of 10 stand / tips |
| Animal-like walk, Froude 0.05 (0.44–0.50 m/s), against the drives | 8 of 10 within limits (knee 0.91–0.98) | 0 of 10: hip pitch 1.07–1.08 (RobStride 03 speed on 24 V), compact knee 1.08–1.11 |
| One-foot stop (spec 1.5 m/s²) | 1.22–1.27 m/s² | 1.19–1.23 m/s² |
| One-step push, CoM speed change (upper bound) | 0.45–0.58 m/s | 0.50–0.55 m/s |
| Hottest drive over a following session (RMS / sustained) | neck 0.76–0.89 | neck 0.76–0.97 |
| Person's torso in view with the head level | all | all |

What this means:

* **Turning works at both sizes.** The swing foot must yaw through a stride's whole turn within its swing
  time, so turning in place needs about 2.7–3.0 rad/s of thigh twist with the margin. The selected drives have
  4.19 rad/s.
  At 60 cm a walking turn asks for friction up to 0.65 in single support, above the carpet band's lower
  edge (0.54): a controller may need to slow turns on slippery carpet.
* **Carpet: the bodies fit, the gait does not yet.** Friction and toe clearance pass everywhere with the
  reference walk, but meeting the carpet swing target needs a swing lift of about 0.17–0.19 H. That is a
  controller change, not a body change. No body stands passively on medium or high pile at 60 cm, so
  standing on thick carpet needs active ankle balance.
* **The 75 cm hip-pitch drive caps the walking speed.** Both sizes meet the spec's nominal and brisk speeds.
  At the animal's preferred Froude 0.05, every 75 cm body exceeds its RobStride 03's speed on the 24 V bus.
  §4 gives the fix and its cost. At 60 cm the bases pass with the knee at 0.96–0.98, so their margin
  is thin too.
* **Stopping hard on one foot needs a step.** No body holds the spec's 1.5 m/s² stop on one foot with
  the ankle alone (1.19–1.27 m/s²), so a controller must stop in double support or take a step.

Tables: `round2/TABLES.md` §8 (turning), §9 (carpet), §10 (behavior envelope).

## 3. Variants: what each single change does

| Variant | 60 cm | 75 cm |
|---|---|---|
| **Narrow stance** (P13) | the narrow width plus 4 mm per side, the least that walks *and* turns without contact (F16); feet 0.36 H apart (base 0.42 H); overall width 270–271 mm (279 mm). Hip-roll hold −11 %, runtime +1.3 min; the tightest packaging (turns clear by 1.2–1.6 mm) | plus 6 mm per side; feet 0.28–0.29 H (0.33–0.34 H); width 265 mm (290–291 mm). Hip-roll hold −10 to −11 %; turning in place clears by only 0.8–1.0 mm |
| **Larger skull** (P13) | skull 23.1–23.2 cm long (base 20.4–20.5 cm). The neck servo rises from 0.76 to 0.85–0.86 of its rating; on Hybrid the Froude 0.05 walk is just over the knee limit (1.00) | 28.8–29.0 cm (25.4–25.6 cm). The heavier head selects the stronger XM540-W270 neck servo, which then runs at 0.76 instead of the base's W150 at 0.96–0.97 |
| **Compact** (P13) | shorter legs and tail: length 141.6 / 136.7 cm (151.0 / 144.9 cm). The centre of mass moves forward (0.13 H ahead of the hip), so the hip-pitch hold rises 35–40 %; Tapered exceeds the knee limit at Froude 0.05 (1.08) | hip-pitch hold +24–27 %; the knee rubbed the calf until the ankle drives moved down to 0.42 of the shank; knee 1.08–1.11 at Froude 0.05 |
| **Level, 0°** (P15) | the same body pitched level: crown 53.5–53.6 cm. Loads, runtime and cost within 1 % of the base, and the scorecard total ties with it (0.70 / 0.68) | crown 66.9–67.0 cm; within 1 % of the base on every load |

The larger-skull bodies are 0.03–0.09 kg *lighter* than their bases. Sizes are matched at the crown, and a
taller head means a slightly smaller body below it (body scale 1.4 % smaller).

## 4. What-if: 75 cm sized for the animal-like walk

`designlab/whatif.py` re-sized `75-M-T-10` and `75-M-H-10` with the Froude 0.05 walk added to the sizing cases
(`round2/whatif_fr05/`). Both stay on 24 V with the same 6S 3.3 Ah pack:

* the hip pitch changes from a RobStride 03 to a MyActuator X6-60, and the thigh twist from a CubeMars AK45-36
  to a MyActuator X4-36-L;
* mass falls by 0.34–0.35 kg (to 14.56 / 14.68 kg), cost rises by $0.8k nominal (to $18.2k), runtime falls by
  1.6 min (to 36.6 / 36.4 min);
* both bodies then meet the Froude 0.05 walk (governing ratio 0.97–0.98) and still pass every round-two gate.
  The worst drive ratio under the stress test falls from 1.39–1.40 to 1.26;
* the scorecard total barely moves (0.54–0.55), because the neck servo and the knee stay within a few per cent
  of their limits.

If a 75 cm body goes to verification, this sizing (or a stronger neck servo as well) is the one to build.
§8 question 3 asks.

## 5. Packaging and the inside

* **Packaging search:** the four narrow stances needed 4 mm (60 cm) or 6 mm (75 cm) per side added back. The
  two 75 cm compact bodies needed their ankle drives lowered to 0.42 of the shank: the thigh touched the
  top of the calf fairing in both walks. Every other body passed at its first build (`round2/TABLES.md` §11).
* **Joint-range margins:** the tightest margin between a walk's extremes and the usable range is 0.13–0.20 rad
  at 60 cm and 0.10–0.18 rad at 75 cm, against the 0.10 rad gate. It is outward hip roll on 17 bodies and knee
  flexion on 3. The narrow stances and the Hybrid bodies have the least margin.
* **Head camera (F8):** the search puts the camera at 0.165–0.170 of the skull length, its box 8–15.5 mm below
  the roof, with 4.2–7.3 % of its view blocked by the robot's own body. Round two sees less of the floor
  ahead than round one did: the nearest floor seen straight ahead is 2.0–2.4 m ahead of the toes at 60 cm and
  2.2–2.6 m at 75 cm (round one: 1.9 and 2.2 m). The two ToF sensors cover 56–73 % of the near foot path. A person's torso is in view at 1.5–3 m with the head level on every body; at the neck's pitch
  limits it is not, so following keeps the head near level (`BEHAVIOR_SPEC.md` §3).
* **Neck:** the S-curved neck brings the head back over the chest, so the 60 cm neck servo (Feetech STS3250)
  drops from 0.97 to 0.76 of its rating. At 75 cm the lighter XM540-W150 now suffices, at 0.96–0.97.

## 6. Scorecard, picks and cross-check

**Method** (`designlab/shortlist.py`, from recorded screen results only):

1. **Gates:** every gate in §1. All 20 pass.
2. **Margins:** each behavior gets a 0–1 score from how far its governing quantity sits from its limit.
3. **Weights** from the owner's priorities: natural walking 0.20, turning 0.20, robustness 0.15,
   following 0.12, carpet 0.10, endurance 0.10, head and tail 0.05, packaging 0.05, cost 0.03.
4. **Stress test:** drive ratios ×1.29 (drives 15 % weaker, robot 10 % heavier).
5. **Diversity:** the best body at each height, then the best remaining body of the other family, so
   verification compares real alternatives.

Looks are the owner's call and are not scored.

| Rank | Body | Total | Natural walking | Turning | Carpet | Robustness | Following | Endurance |
|---|---|---|---|---|---|---|---|---|
| 1 | `60-N-H-10` narrow stance | 0.71 | 0.63 | 0.82 | 0.69 | 0.76 | 0.55 | 0.91 |
| 2 | `60-N-T-10` narrow stance | 0.71 | 0.66 | 0.81 | 0.70 | 0.76 | 0.51 | 0.90 |
| 3 | `60-M-T-00` level | 0.70 | 0.62 | 0.70 | 0.65 | 0.77 | 0.56 | 0.90 |
| 4 | `60-M-T-10` base | 0.70 | 0.63 | 0.70 | 0.66 | 0.76 | 0.52 | 0.90 |
| 5 | `60-M-H-10` base | 0.68 | 0.59 | 0.70 | 0.65 | 0.76 | 0.55 | 0.91 |
| 6 | `60-M-H-00` level | 0.68 | 0.59 | 0.70 | 0.64 | 0.77 | 0.54 | 0.91 |
| 9 | `75-M-T-10-bighead` larger skull | 0.61 | 0.38 | 0.56 | 0.86 | 0.74 | 0.54 | 0.78 |

The full ranking of all 20, with head and tail, packaging, cost and the stressed ratio, is in
`round2/SHORTLIST.md`. Reading it:

* **The six 60 cm base, narrow and level bodies are nearly tied (0.68–0.71).** The narrow stance edges ahead
  on turning: its lower hip-roll and thigh-twist loads leave more drive headroom in the turns. It also has the
  tightest packaging (score 0.17–0.18, against 0.23–0.48 for the base and level bodies).
* **The 75 cm bodies trail (0.55–0.61)** because their drives run at 0.96–0.98 of their limits and the Froude
  0.05 walk exceeds the hip pitch. They lead on carpet (larger feet, more toe clearance). The best 75 cm body
  is the larger skull, mainly because its heavier head selects the stronger neck servo, which then has headroom.
* **Under the stress test every body exceeds a limit** (worst ratio 1.21–1.43). That is expected with ratios
  near 0.8 and a 1.29 stress factor; it marks where verification has the least slack.

**Scorecard picks:** `60-N-H-10` (best at 60 cm), `75-M-T-10-bighead` (best at 75 cm) and `60-N-T-10` (best
remaining body: the two families are already represented).

**Independent cross-check:** _pending; filled in from `round2/CROSSCHECK.md`._

## 7. Corrections to round one

* **The reference walk did not converge (F14).** Round one held the centre of mass to 0.2 mm and estimated the
  stride boundary's derivatives, so the walk's ZMP correction never settled: 27–139 mm of ZMP error remained.
  Re-checking all 132 round-one models at round two's accuracy (`designlab/recheck_r1.py` →
  `round2/round1_recheck.json`, 1.4–3.9 mm):
  * The three walks flagged as leaving the support pad (`60-M-H-20`, `60-W-S-00`, `75-M-S-10-belly`) keep their
    centre of pressure on it. Every nominal walk now has 29–62 mm of margin (round one: −26 to +50 mm), and no
    brisk walk leaves the pad (round one: 44 of 132). The flags were generator artifacts.
  * Continuous requirements, which set most drive choices, change by less than 2 %. Peak requirements fall,
    most at the ankles (to 0.33–1.00 of round one's) and the knee (0.52–1.00). The thigh-twist speed
    requirement falls further, but it was small.
  * Re-selecting drives from the corrected requirements would change at least one part on 25 of 132 bodies:
    an ankle drive on 13 (all 60 cm), the tail servo on 10, the knee on 2 and the hip pitch on 1. The holds and
    continuous loads behind round one's size and family comparisons are unchanged.
* **Turning in place needs several times the thigh-twist speed first stated.** `SCREENING.md` §7 said a 0.5
  rad/s turn in place needs about 0.5 rad/s. It needs about 2.4–2.6 rad/s before the margin at 60 and 75 cm.
  The 18 round-one bodies with a CubeMars AK80-64 twist drive on 24 V (100 and 120 cm) would sit at 1.12 of its
  usable speed (`100-W-T-10`). Corrected in `SCREENING.md` §7.
* `ROUND1_REVIEW.md`, `SCREENING.md` §3 and §7 carry the corrections; round one's generated tables are not
  regenerated.

## 8. Questions for the owner (round two)

1. **Bodies for verification.** The scorecard proposes `60-N-H-10`, `75-M-T-10-bighead` and `60-N-T-10` (§6).
   Accept, swap one (for example on looks), or name a different set.
2. **Priority weights.** Natural walking and turning first, then robustness, following, carpet, endurance,
   head and tail, packaging and cost. Keep them, or reorder?
3. **75 cm drives.** If a 75 cm body goes forward, size it for the Froude 0.05 walk (§4: +$0.8k, −0.3 kg,
   −1.6 min), and also give it a stronger neck servo for headroom? Or keep the brisk-walk sizing?
4. **Calf fairings.** Keep them (an animal's lower leg, tighter knee packaging) or drop them (exposed ankle
   drives and push rods: lighter, easier to service, more robotic)?
5. **Head camera.** Keep it in the skull roof (person in view; the near floor relies on the ToF pair), add a low
   forward camera for the near floor (cost, power and wiring), or move the camera forward and lower (better
   floor view, less of a close person's torso)?
6. **Stage 1 go-ahead.** May the bounded closed-loop probe start on the chosen bodies (§9)?

The answers go into `DECISION_LOG.md` as P-entries.

## 9. Next

* **Stage 1, closed-loop probe** (needs the owner's go-ahead): a bounded controller study on the chosen
  bodies, with the same budget each. It covers walking at the target speeds, the target turns, start and
  stop, and a push, on hard floor and the carpet floors of this round. It is not the final training; it
  checks that a controller can use the margins found here.
* **Stage 2, single-leg bench:** one leg of the leading body. Drive temperatures over a walking duty cycle, the
  knee and ankle clearances, and the measured compute draw (`COMPUTE.md` bench test 1, assumption A14).
* **Verification training:** equal-budget training on the finalists against `BEHAVIOR_SPEC.md`, then one design
  is chosen.

## 10. What this round does not show

* No controller has walked, turned or stood on carpet with any of these models. The turns, walks and stance
  checks are prescribed motions and simple balance models.
* Slip, impacts, touchdown dynamics and carpet damping are not modelled (assumption A16); thermal behaviour is
  the sustained planning fraction only (A1).
* All masses, powers, runtimes and costs are model estimates from sourced part data and stated rules. None is
  measured.
