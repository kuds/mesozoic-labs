# Round-two design review, with the overnight refinement

**Status (8 Nov 2026, morning):** round two plus an overnight refinement session, awaiting the owner's answers to
§12. Nothing here is a final design. No controller training has started (P7), and the proposed next step, a
bounded closed-loop probe, waits for the owner's go-ahead. Every mass, power, runtime and cost is a model estimate
from sourced part data and stated rules; none is measured.

The illustrated version is the round-two review page, published separately and linked in the delivery message
(`python -m designlab.review2_page` builds it from the same records). Tables here are compiled from
`round2/TABLES.md`, `round2/SHORTLIST.md`, `round2/refine/REFINE.md`, `round2/size_study/SIZE_STUDY.md` and
`round2/refine/MATERIALS.md` and, for the refined bodies, from their records under `round2/refine/`
(`candidates/*.json`, `review/review_index.json`, `review/screening/*_turning.json` and the lifted walks in
`carpet_lift_*.json`), rounded for display.

## Summary

**Recommended bodies for verification** (§6; the three picks of scorecard version 2, all refined overnight):

| | Body | What it is | Mass | Runtime | Cost nominal / high | Governing limit |
|---|---|---|---|---|---|---|
| 1 | `60-N-T-10`, narrow, carpet-ready | 60 cm Tapered, narrow stance with the legs moved out 6 mm per side (round two: 4 mm), so every target turn clears by at least 3 mm; ankle drives at 0.60 of the shank | 10.81 kg | 47.6 min | $14.8k / $20.5k | knee speed 0.93 (Froude 0.05 walk) |
| 2 | `75-M-H-10`, targeted drives, bigger feet | 75 cm Hybrid; four drives upgraded (§4); toes ×1.15 and pads ×1.25 (§7.5) | 14.85 kg | 36.7 min | $18.5k / $25.7k | knee speed 0.93 (Froude 0.05 walk) |
| 3 | `60-N-H-10`, narrow, carpet-ready | 60 cm Hybrid, the same layout as 1: the family comparison at 60 cm | 10.91 kg | 47.4 min | $14.8k / $20.6k | knee speed 0.95 (Froude 0.05 walk) |

The top three scorecard totals lie within 0.005 of each other (0.703, 0.702, 0.699), a tie at the scorecard's resolution, so the set rests on what each body adds and on the diversity rule. Comparing 1 and 3 tests the body family at 60 cm on identical layouts; comparing 3 and 2 tests size within the Hybrid family, although the 75 cm body also keeps the base stance and has the bigger feet. The narrow Hybrid's carpet swing is the tightest of the three: it needs 0.195 H to meet the carpet target (the Tapered body 0.19 H), which its nominal walk carries within the drives and every walking gate; with the brisk walk lifted as well its knee-range margin would be 0.097 rad, under the 0.10 rad gate, but its knee's speed rules out a brisk walk with that swing anyway (§7.6). Near alternatives are in §6: the narrow 3 mm bodies without the carpet-ready ankle drives (0.69), the 75 cm faster-knee bodies (0.68), the base-width carpet-ready 60 cm Tapered (0.68) and the targeted 75 cm Tapered body without the bigger feet (0.67).

**What the overnight session found:**

* **The 75 cm body is fixed by four drive changes, not by a bigger robot.** The Froude 0.05 sizing plus 15 %
  headroom on the neck servo and hip roll takes 0.32–0.33 kg off, adds $1.0k nominal, passes every gate and lifts
  the 75 cm bodies' scorecard totals from 0.51–0.52 to 0.66–0.67; it also carries the carpet swing (0.18 H lift)
  within its drives (§4, §7.2).
* **Carpet needed a knee change at 60 cm.** The 60 cm knee could not bend far enough for the carpet swing: the
  thigh met the calf fairing at 0.81 rad. Lowering the ankle drives from 0.46 to 0.60 of the shank opens it to
  1.00–1.05 rad with the fairing kept, without changing a drive (§7.1).
* **Grip, not drive torque, is the 60 cm robot's limit after the knee's speed.** Counting the stance foot's spin,
  the base-width 60 cm bodies need friction 0.54–0.55 just to walk, more than vinyl or laminate may give. The narrow
  stance, now with every turn clearing by 3 mm, needs about 12–14 % less (§7.4, §7.6); longer toes and larger pads
  need about 8 % less and stop harder, but load the ankle drive (§7.5).
* **A smaller 60 cm robot does not gain margin, and 80 cm gains nothing over the targeted 75 cm** (§8).
* **Lighter materials do not move the limit that binds the leading bodies** (the knee's speed in the animal-like
  walk). Thinner MJF PA11 coverings are the best-value material change; carbon-fibre frame plates are not worth
  it for prototype 1, and thinner (2.54 mm) aluminium plate saves 0.16–0.17 kg at no extra cost, under half of 3 mm
  carbon's 0.39–0.44 kg at a similar stiffness (0.61× against 0.67× of today's plate) (§9).
* **Keep the battery, and add an external DC input.** A tether-only robot loses the tail-base counterweight and,
  with the same drives, pushes the knee past its speed limit (re-sized, it needs a faster knee drive). The 3–4 ft robot can run on a battery that is 2–4 % of its mass (§9).
* **The scorecard was corrected (version 2)** after an independent blind cross-check found six flaws in version
  1. A claim audit of 681 checkable claims in the round-two documents had 58 problems reported, 42 confirmed by
  both independent checkers and 15 by one; the corrections are in this review, the decision log, `SCREENING.md`,
  `COMPUTE.md` and `README.md` (§6, §11). A final audit of this report checked 2,232 claims: of 88 reported
  problems, 67 were confirmed by both checkers and 14 by one, and all 81 are corrected; none changes a pick
  (`round2/AUDIT.md`).

**Your overnight questions, short answers:**

1. *Is a 5–10 % smaller 60 cm robot better?* No. It saves only 0.3–0.6 kg, the electronics stop fitting, and at
   54 cm the knee runs out of speed in the animal-like walk (§8).
2. *Can the 75 cm robot take larger motors, or does it need 80 cm?* Yes, at 75 cm: four targeted drive changes
   (stronger hip roll and thigh twist, a faster hip pitch, the stronger neck servo) pass every gate and even take
   0.32–0.33 kg off. 80 cm is not needed and is no better (§4, §8).
3. *Head and tail movement?* One nodding neck joint (±20°) and one side-to-side tail joint (±26°); in the walk the
   head nods ±3° and the tail sways ±7° (§10).
4. *Materials, structural and cosmetic?* Keep 6061-T6 aluminium plate for the frame (2.54 mm where a leg bench test
   allows); PA11 coverings with graded, thinner walls; polyurethane foot-pad treads, measured on your floors; foam
   bumpers at strike points (§9).
5. *Carbon fibre and other light, strong materials?* 3 mm carbon plate saves about 0.4 kg for $1.6–2.1k, but it
   is only two-thirds as stiff as today's aluminium; at equal stiffness (the 3.99 mm stock gauge; 3.43 mm is needed
   but not stocked) it saves 0.20–0.22 kg, while thinner aluminium plate saves 0.16–0.17 kg for nothing. Carbon also hides fall damage; titanium gains nothing on plates;
   magnesium saves about a quarter of the plate mass but corrodes and few shops cut it. A hybrid carbon and
   aluminium frame is worth revisiting for the 3–4 ft robot (§9).
6. *Non-battery version?* Not tether-only; battery plus an external DC input (§9).
7. *Battery on the 3–4 ft build?* Yes: about 50–94 min of walking from a 0.9–1.7 kg pack (§9).

## How round two was run

1. **Candidates (P10–P15):** 60 and 75 cm crown height × Tapered and Hybrid bodies. Each base body
   (balanced width, 10° chest-up) has four single-change variants: narrow stance, larger skull, compact
   proportions (shorter legs, tail and neck, larger skull) and the level 0° posture. The level variant is the
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
   stress test and a diversity rule (`designlab/shortlist.py`). A blind three-judge panel and an adversarial
   comparator cross-checked the first scorecard's picks (`round2/CROSSCHECK.md`). Their findings produced
   scorecard version 2 (F18), which ranks the bodies in §6.
7. **Overnight refinement (§7–§9):** the size study and single-lever experiments on the leading bodies, sized and
   screened like the 20 candidates, and a materials and power study (every lever rebuilt with the reference body's
   drives and pack, without re-screening; the 2 mm carbon, PA12 thin-wall, combined and tether levers also re-sized
   and screened). §10 describes head and tail movement.

## 1. Results at a glance (the 20 round-two candidates)

All 20 candidates pass every round-two gate of the first scorecard:

* every clearance gate: no contact at home or in either walk, no overlapping parts, swing-foot and tail
  clearance, a battery removal path and the 0.10 rad joint-range margin;
* every drive within its planning limits in the sizing walks and the target turns, with IK holding and no
  contact in the turns;
* a person's torso in view, at least 20 min of runtime, the carpet gate;
* the nominal reference walk's centre of pressure within 10 mm of the pad in single support (all are 29–38 mm
  inside it).

Scorecard version 2 adds one gate, the Froude 0.05 walk within the drives (F18). Twelve of the 20 fail it:
all ten 75 cm bodies, `60-M-T-10-compact` and `60-M-H-10-bighead`.

| Base body | Crown / hip axis / length | Mass | Walking power | Runtime | Cost nominal / high | Governing drive ratio |
|---|---|---|---|---|---|---|
| `60-M-T-10` Tapered | 60.1 / 40.5 / 151.0 cm | 10.85 kg | 71 W | 46.4 min | $14.8k / $20.6k | thigh twist, sustained 0.77 |
| `60-M-H-10` Hybrid | 60.1 / 40.1 / 144.9 cm | 10.96 kg | 71 W | 46.2 min | $14.9k / $20.7k | knee speed 0.79 |
| `75-M-T-10` Tapered | 75.1 / 50.6 / 183.7 cm | 14.90 kg | 86 W | 38.2 min | $17.4k / $24.5k | neck servo, sustained 0.97 |
| `75-M-H-10` Hybrid | 75.1 / 50.2 / 175.9 cm | 15.03 kg | 87 W | 38.0 min | $17.5k / $24.6k | thigh twist, sustained 0.97 |

Every candidate runs on 24 V with a 6S 3.3 Ah pack. Against the same four round-one ids, walking power falls
by 22–25 W, mostly from the Pi 5 compute (9.5 W budget against the Jetson's 31 W), so runtime rises from
30–34 min to 38–46 min. Mass changes by +0.08 to +0.20 kg and nominal cost by less than $70 (the high-side
estimate rises by $84–204).

**The 60 cm base, narrow and level bodies keep about 20 % drive headroom; the 75 cm bodies do not.** At 60 cm
the governing ratio is 0.77–0.79 on the base, narrow and level bodies, and 0.85–0.96 on the larger-skull and
compact variants. At 75 cm it is 0.96–0.98 on every body. On seven of them that is the thigh twist (CubeMars
AK45-36) at its sustained limit. On three it is the neck servo (ROBOTIS XM540-W150, 0.96–0.97 wherever it is
selected); the larger-skull and compact bodies select the stronger XM540-W270, which then runs at 0.76–0.80.

## 2. Behaviors (Stage 0 screens)

These checks use prescribed motions and simple balance models against the selected drives with the sizing
margins. For the behavior-envelope rows (Froude 0.05 walk, one-foot stop, push, session heat), failing one
shows that the prescribed motion exceeds that body and those drives at those margins; passing one does not
show that a controller can produce the behavior (`SCREENING.md` §8). The carpet swing target and the
servo-held stance test the reference walk's lift and a stance with no balance control, so failing them shows
what a controller must add.

| Behavior | 60 cm (10 bodies) | 75 cm (10 bodies) |
|---|---|---|
| Target turns (0.35 rad/s walking, 0.5 rad/s in place), drives / IK / contact | all pass; governing ratio 0.77–0.96 | all pass; 0.96–0.98 |
| Thigh-twist speed for the turn in place (with the 1.15 margin, of 4.19 rad/s usable) | 2.70–2.81 rad/s | 2.87–2.97 rad/s |
| Friction needed in a walking turn, single support (yaw moment included) | μ 0.53–0.65 | μ 0.37–0.43 |
| Same measure, straight walks (nominal / brisk) | 0.47–0.57 / 0.73–0.88 | 0.33–0.38 / 0.51–0.59 |
| Carpet gate (force ratio below 0.54, yaw moment not included; swing foot above the pile top) | all pass; need 0.24–0.28, toe ≥ 5.3 mm above high pile | all pass; need 0.21–0.24, toe ≥ 10.1 mm |
| Carpet swing target (0.06 H and at least 15 mm above the pile, not gated) | none with the reference lift (0.10–0.11 H); needs about 0.19 H | none; needs 0.17–0.18 H |
| Servo-held stance with no balance control: low / medium / high pile | stands / tips / tips | stands / 4 of 10 stand / tips |
| Animal-like walk, Froude 0.05 (0.44–0.50 m/s), against the drives | 8 of 10 within limits (knee 0.91–0.98) | 0 of 10: hip pitch 1.05–1.08 on every body (RobStride 03 planning speed on 24 V); on the two compacts the knee is worse, 1.08–1.11 |
| One-foot stop (spec limit 1.5 m/s²) | holds 1.23–1.27 m/s² | 1.19–1.23 m/s² |
| One-step push, CoM speed change (upper bound) | 0.45–0.58 m/s | 0.50–0.55 m/s |
| Hottest drive over a following session (RMS / sustained) | neck 0.76–0.89 | neck 0.76–0.97 |
| Person's torso in view with the head level | all | all |

What this means:

* **Turning works at both sizes.** The swing foot must yaw through a stride's whole turn within its swing
  time, so turning in place needs about 2.7–3.0 rad/s of thigh twist with the margin. The selected drives have
  4.19 rad/s.
* **Grip, not turning, is the 60 cm weak spot.** With the stance foot's yaw moment counted, the 0.35 rad/s
  walking turn needs friction up to 0.65 at 60 cm, above the carpet band's lower edge (0.54). The straight walks
  already need 0.47–0.57 (nominal) and 0.73–0.88 (brisk) by that measure, and 0.51–0.59 at 75 cm brisk. The yaw
  moment a small pad must carry, more than the turn, sets the need (F21). A controller may need to walk and turn
  more gently on slippery floors; §9 covers pad compounds.
* **Carpet: the bodies fit; the gait and the knee range needed work.** Toe clearance passes everywhere with the
  reference walk, and so does friction as a force ratio. Meeting the carpet swing target needs a lift of about
  0.17–0.19 H. On the round-two bodies the knee runs out of range before that lift at both sizes (the thigh meets
  the calf fairing at 0.71–0.81 rad; only the two 75 cm compacts reach 0.95 rad). §7.1 gives the 60 cm fix found
  overnight; with the targeted drives (§4) the 75 cm knee opens to 0.95–1.00 rad and both walks meet the target at
  0.18 H (§7.2). No body stands passively on medium or high pile at 60 cm, so standing on
  thick carpet needs active ankle balance.
* **The 75 cm hip-pitch drive caps the walking speed as sized.** Both sizes meet the spec's nominal and brisk
  speeds. At the animal's preferred Froude 0.05, every 75 cm body exceeds its RobStride 03's planning speed on
  the 24 V bus (the walk's peak is 0.91–0.94 of the drive's speed, 1.05–1.08 with the 1.15 margin). §4 gives the
  fix and its cost.
* **A hard stop needs both feet down or a step.** No body can decelerate at the spec's 1.5 m/s² limit on one
  foot with the ankle alone (it holds 1.19–1.27 m/s²), so a stop at that limit needs double support or a step.
  Gentler stops can be held on one foot.

Tables: `round2/TABLES.md` §8 (turning), §9 (carpet), §10 (behavior envelope).

## 3. Variants: what each single change does

| Variant | 60 cm | 75 cm |
|---|---|---|
| **Narrow stance** (P13) | the narrow width plus 4 mm per side, the least that walks *and* turns without contact (F16); feet 0.36 H apart (base 0.42 H); overall width 270–271 mm (279 mm). Hip-roll hold −11 %, runtime +1.3 min; the tightest packaging (turns clear by 1.2–1.6 mm) | plus 6 mm per side; feet 0.28–0.29 H (0.33–0.34 H); width 265 mm (290–291 mm). Hip-roll hold −10 to −11 %; turning in place clears by only 0.8–1.0 mm |
| **Larger skull** (P13) | skull 23.1–23.2 cm long (base 20.4–20.5 cm). The neck servo rises from 0.76 to 0.85–0.86 of its rating; on Hybrid the Froude 0.05 walk is just over the knee limit (1.00) | 28.8–29.0 cm (25.4–25.6 cm). The heavier head selects the stronger XM540-W270 neck servo, which then runs at 0.76 instead of the base's W150 at 0.96–0.97 |
| **Compact** (P13) | shorter legs, tail and neck: length 141.6 / 136.7 cm (151.0 / 144.9 cm). The centre of mass moves forward (0.12–0.13 H ahead of the hip, against 0.09–0.10 H on the base), so the hip-pitch hold rises 35–40 %; Tapered exceeds the knee limit at Froude 0.05 (1.08) | hip-pitch hold +24–27 %; the thigh met the top of the calf fairing as the knee flexed until the ankle drives moved down to 0.42 of the tibia (Tapered: in the brisk walk at 0.30, then only inside the 0.10 rad knee-flexion margin at 0.34 and 0.38; Hybrid: in both walks at 0.30 and in the brisk walk at 0.34 and 0.38); knee 1.08–1.11 at Froude 0.05 |
| **Level, 0°** (P15) | the same body pitched level: crown 53.5–53.6 cm. Leg-drive holds and continuous loads, runtime and cost within 1 % of the base (the tail servo's load is 4 % higher; peak and speed requirements within 2.4 %) | crown 66.9–67.0 cm; leg-drive continuous loads within 1 % of the base except the Tapered knee (hold −1.8 %); the tail servo's load is 5 % higher, and peak and speed requirements within 2.5 % |

The larger-skull bodies are 0.03–0.09 kg *lighter* than their bases. Sizes are matched at the crown, and a
taller head means a slightly smaller body below it (body scale 1.4–1.5 % smaller).

## 4. 75 cm drives: from the what-if to the targeted upgrade

**What-if (F15).** `designlab/whatif.py` re-sized `75-M-T-10` and `75-M-H-10` with the Froude 0.05 walk added to
the sizing cases (`round2/whatif_fr05/`). Both stay on 24 V with the same 6S 3.3 Ah pack. The hip pitch changes
from a RobStride 03 to a MyActuator X6-60 and the thigh twist from a CubeMars AK45-36 to a MyActuator X4-36-L.
Mass falls by 0.34–0.35 kg, cost rises by $0.8k nominal and runtime falls by 1.5–1.6 min. Both bodies then meet
the Froude 0.05 walk and pass every gate, but the scorecard total barely moves (version 2: 0.53, against 0.51–0.52
as sized): the neck servo, the
knee and the hip roll stay within a few per cent of their limits.

**Targeted upgrade (overnight, F19).** Every cross-check judge wanted the what-if drives plus the stronger
neck servo. `round2/refine/targeted_75` sizes both bodies with the Froude 0.05 walk and 15 % headroom on the two
drives every judge found thin, the neck servo and the hip roll:

| Body | Drives that change against round two | Mass | Runtime | Cost nominal / high | Governing ratio | Froude 0.05 |
|---|---|---|---|---|---|---|
| `75-M-T-10` | hip roll AK45-36 → X4-36-L; hip pitch RobStride 03 → X6-60; thigh twist AK45-36 → X4-36-L; neck XM540-W150 → W270 | 14.58 kg (14.90) | 37.6 min (38.2) | $18.4k / $25.4k ($17.4k / $24.5k) | knee speed 0.95 | within (knee 0.95) |
| `75-M-H-10` | the same four | 14.71 kg (15.03) | 37.3 min (38.0) | $18.4k / $25.5k ($17.5k / $24.6k) | knee speed 0.97 | within (knee 0.97) |

The neck servo falls from 0.96–0.97 to 0.66–0.67 of its rating and the hip roll from 0.93 to 0.73. Both bodies pass
every version-2 gate, and their scorecard totals rise from 0.51–0.52 to 0.66–0.67 (§6). The knee's speed in the
Froude 0.05 walk is now the governing limit (0.95–0.97). It does not need a body change, but it is the drive
with the least margin; §7 tests a faster knee.

## 5. Packaging and the inside

* **Packaging search:** the four narrow stances needed 4 mm (60 cm) or 6 mm (75 cm) per side added back. The
  two 75 cm compact bodies needed their ankle drives lowered to 0.42 of the tibia (knee to hock): the thigh
  touched the top of the calf fairing (Hybrid in both walks, Tapered in the brisk walk), and the knee-flexion
  margin to that contact stayed under the 0.10 rad gate until 0.42. Every other body passed at its first build
  (`round2/TABLES.md` §11).
* **Joint-range margins:** the tightest margin between a walk's extremes and the usable range is 0.13–0.20 rad
  at 60 cm and 0.10–0.18 rad at 75 cm, against the 0.10 rad gate. It is outward hip roll on 17 bodies and knee
  flexion on 3. At 60 cm the narrow stances (0.13 rad) and the Hybrid bodies (0.15 rad) have the least margin;
  at 75 cm the four Hybrid bodies other than the narrow stance do (0.10–0.13 rad), and the narrow stances sit
  mid-range (0.16 rad).
* **Head camera (F8):** the search puts the camera at 0.165–0.170 of the skull length, its box 8–15.5 mm below
  the roof, with 4.2–7.3 % of its view blocked by the robot's own body. The nearest floor seen straight ahead is
  2.0–2.4 m ahead of the toes at 60 cm and 2.2–2.6 m at 75 cm (round one, same ids: 2.1 m for `60-M-T-10` and
  2.0 m for `60-M-H-10`; no 75 cm Tapered or Hybrid body was sensor-screened in round one). The two ToF sensors
  cover 56–73 % of the near foot path. A person's torso is in view at 1.5–3 m with the head level on every body.
  At the neck's pitch limits it is not (except with the head fully up on `75-M-T-10-compact`, 77 % visible), so
  following keeps the head near level (`BEHAVIOR_SPEC.md` §3).
* **Neck:** the S-curved neck brings the head back over the chest, so the 60 cm neck servo (Feetech STS3250)
  runs at 0.76 of its rating (round one: 0.97). At 75 cm the lower-torque XM540-W150 (the same 165 g and price
  as the W270 in the catalog) suffices on the base, narrow and level bodies, at 0.96–0.97; the targeted upgrade
  returns to the W270 for headroom.

## 6. Scorecard, cross-check and picks

**Independent cross-check (`round2/CROSSCHECK.md`).** Three judges, each with one lens (behavior risk, hardware
engineering, the owner's stated priorities), chose three bodies blind to the scorecard; an adversarial comparator
then checked every disagreement against the records.

| Who | Picks |
|---|---|
| Scorecard, version 1 | `60-N-H-10`, `75-M-T-10-bighead`, `60-N-T-10` |
| Judge: behavior risk | `75-M-T-10`, `75-M-H-10`, `60-M-T-10` |
| Judge: hardware | `60-M-T-10`, `60-M-H-10`, `75-M-T-10` |
| Judge: owner priorities | `60-M-T-10`, `60-N-T-10`, `75-M-T-10` |
| Comparator's recommendation | `60-M-T-10`, `75-M-T-10` (what-if drives plus the stronger neck servo), `60-M-H-10` |

Every judge's `75-M-T-10` was the what-if body with the stronger XM540-W270 neck servo added, which had not been
sized. The comparator found six flaws in scorecard version 1. It counted the 75 cm neck servo in four
categories, so one part choice decided the 75 cm pick. It reported a stress test it never used. It scored
packaging without the turn clearances, where the narrow stances carry their risk. One sub-score was constant on
every body. It did not require the animal-like walk, and it could not score the what-if bodies.

**Scorecard version 2 (F18)** fixes all six (`designlab/shortlist.py`):

1. **Gates:** every round-two gate, plus the Froude 0.05 walk within the drives (new).
2. **Margins:** each behavior scores 0–1 by how far its governing quantity sits from its limit; each drive group
   counts only in its own categories: the neck and tail servos in head and tail, the leg drives in walking,
   turning, endurance and (through the stress test, item 4) robustness.
3. **Weights** from the owner's priorities, unchanged: natural walking 0.20, turning 0.20, robustness 0.15,
   following 0.12, carpet 0.10, endurance 0.10, head and tail 0.05, packaging 0.05, cost 0.03.
4. **Stress test**, now part of the robustness score: torque ratios ×1.10/0.85 (robot 10 % heavier, drives 15 %
   weaker), speed ratios ÷0.85.
5. **Shortlist:** the best passing body at each height, then the other family's counterpart of the best overall.
6. It ranks the what-if, size-study and refinement bodies alongside the 20 candidates (`--extra`).

**Version 2 ranking** (`round2/SHORTLIST.md`; the 20 candidates plus the what-if bodies and the §7 refinement bodies, 38 of 56 passing every gate; the top 14 shown). The eight material and power bodies of §9 are scored in `REFINE.md` but not ranked here; three of them, all on the base-width carpet-ready `60-M-T-10`, would place 4th, 6th and 8th without changing the picks: 2 mm carbon plates (the optimistic floor) with PA12 walls ×0.75 0.695, PA12 walls ×0.75 0.690, 2 mm carbon plates 0.688:

| Rank | Body | Total | Walk | Turn | Carpet | Robust | Follow | Endur. | Head/tail | Pack. | Cost | Stressed worst |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `60-N-T-10 [refine/narrow_carpet_60]` | 0.703 | 0.51 | 0.81 | 0.70 | 0.64 | 0.51 | 1.00 | 0.86 | 0.76 | 1.00 | 1.09 |
| 2 | `75-M-H-10 [refine/feet_75]` | 0.702 | 0.50 | 0.78 | 0.87 | 0.69 | 0.58 | 0.83 | 0.90 | 0.78 | 0.65 | 1.10 |
| 3 | `60-N-H-10 [refine/narrow_carpet_60]` | 0.699 | 0.48 | 0.82 | 0.69 | 0.62 | 0.55 | 1.00 | 0.88 | 0.75 | 1.00 | 1.12 |
| 4 | `60-N-T-10 [refine/narrow_3mm]` | 0.694 | 0.50 | 0.80 | 0.69 | 0.63 | 0.51 | 1.00 | 0.86 | 0.76 | 1.00 | 1.10 |
| 5 | `60-N-H-10 [refine/narrow_3mm]` | 0.689 | 0.45 | 0.81 | 0.68 | 0.61 | 0.55 | 1.00 | 0.88 | 0.75 | 1.00 | 1.13 |
| 6 | `75-M-H-10 [refine/knee_75]` | 0.682 | 0.56 | 0.69 | 0.86 | 0.68 | 0.58 | 0.73 | 0.89 | 0.76 | 0.65 | 1.09 |
| 7 | `75-M-T-10 [refine/knee_75]` | 0.675 | 0.57 | 0.69 | 0.87 | 0.69 | 0.54 | 0.73 | 0.77 | 0.78 | 0.66 | 1.08 |
| 8 | `60-M-T-10 [refine/carpet_60]` | 0.675 | 0.47 | 0.72 | 0.66 | 0.62 | 0.52 | 1.00 | 0.86 | 0.83 | 1.00 | 1.12 |
| 9 | `60-N-T-10` | 0.674 | 0.49 | 0.81 | 0.70 | 0.63 | 0.51 | 1.00 | 0.86 | 0.30 | 1.00 | 1.10 |
| 10 | `60-N-H-10 [refine/no_calf]` | 0.673 | 0.45 | 0.85 | 0.69 | 0.60 | 0.55 | 1.00 | 0.88 | 0.29 | 1.00 | 1.13 |
| 11 | `75-M-T-10 [refine/targeted_75]` | 0.671 | 0.46 | 0.76 | 0.87 | 0.61 | 0.53 | 0.85 | 0.78 | 0.78 | 0.66 | 1.11 |
| 12 | `60-M-T-10 [refine/no_calf]` | 0.669 | 0.45 | 0.73 | 0.66 | 0.61 | 0.52 | 1.00 | 0.86 | 0.83 | 1.00 | 1.13 |
| 13 | `60-N-H-10` | 0.669 | 0.45 | 0.82 | 0.69 | 0.61 | 0.55 | 1.00 | 0.88 | 0.29 | 1.00 | 1.13 |
| 14 | `60-N-H-10 [refine/pads_large]` | 0.666 | 0.44 | 0.83 | 0.69 | 0.63 | 0.55 | 1.00 | 0.88 | 0.18 | 1.00 | 1.13 |

Reading the ranking:

* **The top three are the picks, and they tie.** Their totals (0.703, 0.702, 0.699) lie within 0.005, below the
  scorecard's resolution, so the set rests on the shortlist rule (the best passing body at each height, then the
  other family's counterpart of the best overall) and on what each body adds (Summary).
* **The narrow carpet-ready 60 cm bodies lead on turning and packaging together.** They keep the narrow stance's
  turning score (0.81–0.82), and with every target turn clearing by at least 3 mm their packaging score is 0.75–0.76,
  against 0.29–0.30 for the round-two narrow bodies, whose turns cleared by 1.2–1.6 mm (§7.4, §7.6).
* **The 75 cm Hybrid with bigger feet is the best 75 cm body.** The bigger feet add to the targeted drives the
  hardest one-foot stop of the three picks (robustness 0.69, against 0.59 without them) and more grip and knee-speed
  margin (natural walking 0.50 against 0.41, turning 0.78 against 0.75). Its carpet score is 0.87 with or without
  them (it scores toe clearance and friction margin, not standing), although only with them does it stand on
  medium pile with its servos held. Its cost score (0.65) is the
  price of the larger robot (§7.5).
* **Family:** at 60 cm the Tapered and Hybrid bodies are 0.004 apart on the narrow stance and 0.017 at base width.
  Tapered leads on natural walking (0.51 against 0.48 on the narrow stance), Hybrid on following (0.55 against 0.51).
* **Near alternatives:** the narrow 3 mm bodies without the carpet-ready ankle drives (0.69, ranks 4 and 5); the
  75 cm faster-knee bodies (0.68, ranks 6 and 7, §7.3); the base-width carpet-ready 60 cm Tapered (0.68, rank 8),
  which led the provisional overnight recommendation; the targeted 75 cm Tapered without the feet (0.67, rank 11).
* **No body passes the stress case outright.** With the robot 10 % heavier and the drives 15 % weaker and slower,
  each pick's knee would run 9–12 % over its speed limit in the Froude 0.05 walk ("Stressed worst"); the test
  grades how far over, inside the robustness score, and is not a gate.

## 7. Overnight refinement: one lever at a time

`designlab/refine.py` changes one lever on a leading body, re-sizes it with the round-two rules (generator
revision 2, Pi 5 compute, round-two walk accuracy, the packaging search) and screens it exactly like the 20
candidates (`round2/refine/REFINE.md`; one directory per case). No controller is involved.

### 7.1 Carpet-ready 60 cm (F20)

The carpet swing target (0.06 H above the pile top, at least 15 mm) needs a swing lift of about 0.19 H at 60 cm.
On the round-two 60 cm base bodies the knee stops at 0.81 rad (0.76–0.81 rad across the ten 60 cm bodies), where
the upper-thigh shell meets the calf fairing, and
a 0.19 H swing needs about 1.0 rad. Lowering the ankle drives from 0.46 to 0.60 of the shank moves the fairing
down and opens the knee to 1.05 rad (Tapered) and 1.00 rad (Hybrid) with the fairing kept. Sized with the nominal
walk lifted to 0.19 H and the Froude 0.05 walk as extra sizing walks (`round2/refine/carpet_60`), neither body
changes a drive:

| Body | Mass | Runtime | Cost nominal / high | Governing limit | Target turns | Tightest joint margin | Score (v2) | Gates |
|---|---|---|---|---|---|---|---|---|
| `60-M-T-10` carpet-ready | 10.83 kg (10.85) | 46.5 min (46.4) | $14.8k / $20.6k | knee speed 0.95 (Froude 0.05) | pass, 0.76 | 0.20 rad | 0.68 (0.66) | all pass |
| `60-M-H-10` carpet-ready | 10.93 kg (10.96) | 46.3 min (46.2) | $14.9k / $20.7k | knee speed 0.97 (Froude 0.05) | pass, 0.77 | 0.15 rad | 0.66 (0.65) | all pass |

Round-two values in brackets. The knee's speed in the Froude 0.05 walk was already 0.96–0.98 on these bodies; the
lower ankle drives change the leg's mass distribution slightly and it becomes 0.95–0.97.

With both reference walks lifted (`round2/refine/carpet_lift_carpet_60*.json`):

| Lift | Lowest point above a high-pile top (target 24.3 / 24.1 mm) | Drives | Walking clearance gates |
|---|---|---|---|
| 0.16 H | 18.3 / 18.0 mm | within (knee 0.96–0.97) | pass |
| 0.19 H | 24.3 / 24.0 mm: Tapered meets the target, Hybrid 0.05 mm short | nominal walk within (knee 0.85–0.87); brisk walk over (1.06–1.08) | pass |
| 0.195 H | 25.4 / 25.0 mm: both meet the target | nominal walk within (knee 0.87–0.88); brisk walk over (1.08–1.09) | pass |

So on carpet the 60 cm robot walks at the nominal speed with a 0.195 H swing: with the nominal walk alone lifted,
both bodies meet the target, every drive stays within its limits (the knee at 0.87–0.88 of its speed limit) and
every walking gate passes (`round2/refine/carpet_lift_carpet_60_0.190_0.195_nominal_only.json`); a brisk walk
with that swing exceeds the knee's speed.

### 7.2 Targeted 75 cm and the carpet swing

§4 gives the targeted upgrade. Sized again with the nominal walk lifted to 0.18 H as an extra sizing walk
(`carpet_75`), the 75 cm bodies select exactly the same drives (the same model, SHA-256 identical), so the
targeted body already carries the carpet swing. With both reference walks lifted
(`carpet_lift_targeted_75.json` and `carpet_lift_targeted_75_0.175_0.180.json`):

| Lift | `75-M-T-10`: above a high-pile top (target 30.4 mm) | `75-M-H-10` (target 30.1 mm) | Drives (knee) | Walking clearance gates |
|---|---|---|---|---|
| 0.16 H | 26.4 mm | 26.1 mm | within (0.91 / 0.93) | pass |
| 0.175 H | 30.2 mm | 29.8 mm | within (0.96 / 0.97) | pass |
| 0.18 H | 31.4 mm: met | 31.1 mm: met | within (0.97 / 0.99) | pass |
| 0.19 H | 34.0 mm | 33.6 mm | over: Tapered knee 1.005, Hybrid 1.02 | Tapered pass; Hybrid knee margin 0.09 rad, under the 0.10 rad gate |

### 7.3 A faster knee (15 % knee headroom)

On the refined bodies the governing limit is the knee's speed (0.95 Tapered, 0.97 Hybrid). With 15 % headroom on the knee the sizing
selects a RobStride 06 knee (0.621 kg, against the MyActuator X4-36-L's 0.36 kg) and the knee's speed falls to
0.55–0.57 of its limit:

| Body | Mass | Runtime | Cost nominal | Knee speed | Governing limit after | Target turns | Friction need (turn) |
|---|---|---|---|---|---|---|---|
| `60-M-T-10` | 11.57 kg (10.83) | 40.1 min (46.5) | $14.8k ($14.8k) | 0.56 (0.95) | hip pitch speed 0.90 | 0.83 (0.76) | 0.73 (0.63) |
| `60-M-H-10` | 11.67 kg (10.93) | 40.2 min (46.3) | $14.9k ($14.9k) | 0.57 (0.97) | hip pitch speed 0.90 | 0.84 (0.77) | 0.71 (0.62) |
| `75-M-T-10` | 15.43 kg (14.58) | 31.4 min (37.6) | $18.4k ($18.4k) | 0.55 (0.95) | tail servo peak 0.91 | 0.83 (0.78) | 0.51 (0.46) |
| `75-M-H-10` | 15.55 kg (14.71) | 31.5 min (37.3) | $18.5k ($18.4k) | 0.57 (0.97) | thigh twist sustained 0.84 | 0.84 (0.78) | 0.49 (0.45) |

The reference body's value is in brackets (`carpet_60` or `targeted_75`).

The heavier legs (0.74–0.85 kg more in all) cost turning headroom (0.83–0.84 against 0.76–0.78), grip (friction
need 0.71–0.73 at 60 cm and 0.49–0.51 at 75 cm, against 0.62–0.63 and 0.45–0.46) and about 6 min of runtime. The runtime part rests on the model's fallback winding
resistance, because RobStride does not publish the RobStride 06's. The scorecard puts the 75 cm faster-knee bodies
level with or above the targeted ones (Tapered 0.675 against 0.671, Hybrid 0.682 against 0.664) and the 60 cm
faster-knee bodies below the carpet-ready ones (§6). Not adopted as the
default; question 3 offers it at 75 cm.

### 7.4 Feet, calf fairing, narrow stance and blanket headroom

| Body | Lever | Mass | Governing drive | Target turns | Friction need (turn) | One-foot stop | Push | Tightest joint margin | Score | Gates |
|---|---|---|---|---|---|---|---|---|---|---|
| `60-M-T-10` | round two (reference) | 10.85 kg | thigh twist sustained 0.77 | 0.77 | 0.65 | 1.24 m/s² | 0.49 m/s | 0.20 rad | 0.66 | all pass |
| `60-M-H-10` | round two (reference) | 10.96 kg | knee speed 0.79 | 0.78 | 0.63 | 1.24 m/s² | 0.48 m/s | 0.15 rad | 0.65 | all pass |
| `60-N-H-10` | round two (reference) | 10.93 kg | knee speed 0.77 | 0.77 | 0.53 | 1.24 m/s² | 0.48 m/s | 0.13 rad | 0.67 | all pass |
| `60-N-T-10` | round two (reference) | 10.84 kg | ankle A sustained 0.77 | 0.77 | 0.54 | 1.24 m/s² | 0.48 m/s | 0.14 rad | 0.67 | all pass |
| `75-M-T-10` | round two (reference) | 14.90 kg | neck sustained 0.97 | 0.97 | 0.43 | 1.20 m/s² | 0.55 m/s | 0.18 rad | 0.52 | **fails**: animal walk |
| `60-M-T-10` | 60 cm carpet-ready + toes x1.15 and pads x1.25 | 10.92 kg | knee speed 0.98 | 0.87 | 0.58 | 1.46 m/s² | 0.54 m/s | 0.19 rad | 0.63 | all pass |
| `60-M-H-10` | 60 cm carpet-ready + toes x1.15 and pads x1.25 | 11.02 kg | knee speed 0.96 | 0.87 | 0.57 | 1.46 m/s² | 0.53 m/s | 0.15 rad | 0.64 | all pass |
| `75-M-T-10` | 75 cm targeted + toes x1.15 and pads x1.25 | 14.71 kg | thigh twist sustained 1.00 | 1.00 | 0.42 | 1.41 m/s² | 0.60 m/s | 0.17 rad | 0.55 | all pass |
| `75-M-H-10` | 75 cm targeted + toes x1.15 and pads x1.25 | 14.85 kg | knee speed 0.93 | 0.77 | 0.41 | 1.41 m/s² | 0.59 m/s | 0.17 rad | 0.70 | all pass |
| `60-M-T-10` | no calf fairing (exposed ankle drives) | 10.72 kg | knee speed 0.77 | 0.76 | 0.63 | 1.23 m/s² | 0.48 m/s | 0.20 rad | 0.67 | all pass |
| `60-M-H-10` | no calf fairing (exposed ankle drives) | 10.82 kg | knee speed 0.79 | 0.77 | 0.62 | 1.23 m/s² | 0.48 m/s | 0.15 rad | 0.65 | all pass |
| `60-N-H-10` | no calf fairing (exposed ankle drives) | 10.79 kg | knee speed 0.77 | 0.76 | 0.52 | 1.24 m/s² | 0.47 m/s | 0.14 rad | 0.67 | all pass |
| `75-M-T-10` | no calf fairing (exposed ankle drives) | 14.69 kg | neck sustained 0.97 | 0.97 | 0.42 | 1.20 m/s² | 0.54 m/s | 0.18 rad | 0.53 | **fails**: animal walk |
| `60-M-T-10` | pads x1.25 | 10.91 kg | thigh twist sustained 0.78 | 0.78 | 0.64 | 1.33 m/s² | 0.50 m/s | 0.20 rad | 0.66 | all pass |
| `60-M-H-10` | pads x1.25 | 11.01 kg | knee speed 0.79 | 0.79 | 0.63 | 1.33 m/s² | 0.49 m/s | 0.15 rad | 0.65 | all pass |
| `60-N-H-10` | pads x1.25 | 10.99 kg | knee speed 0.77 | 0.76 | 0.53 | 1.33 m/s² | 0.49 m/s | 0.13 rad | 0.67 | all pass |
| `60-M-T-10` | toes x1.15 | 10.89 kg | ankle A sustained 0.88 | 0.88 | 0.59 | 1.36 m/s² | 0.52 m/s | 0.20 rad | 0.62 | all pass |
| `60-M-H-10` | toes x1.15 | 10.99 kg | ankle A sustained 0.88 | 0.88 | 0.58 | 1.36 m/s² | 0.51 m/s | 0.15 rad | 0.62 | all pass |
| `60-N-H-10` | toes x1.15 | 10.96 kg | ankle A sustained 0.89 | 0.89 | 0.48 | 1.37 m/s² | 0.51 m/s | 0.13 rad | 0.62 | all pass |
| `60-M-T-10` | toes x1.15 + pads x1.25 | 10.94 kg | ankle A sustained 0.87 | 0.87 | 0.59 | 1.45 m/s² | 0.53 m/s | 0.20 rad | 0.63 | all pass |
| `60-M-H-10` | toes x1.15 + pads x1.25 | 11.04 kg | ankle A sustained 0.88 | 0.88 | 0.58 | 1.45 m/s² | 0.52 m/s | 0.15 rad | 0.64 | all pass |
| `60-N-H-10` | toes x1.15 + pads x1.25 | 11.02 kg | ankle A sustained 0.88 | 0.88 | 0.48 | 1.46 m/s² | 0.52 m/s | 0.13 rad | 0.64 | all pass |
| `75-M-T-10` | toes x1.15 + pads x1.25 | 15.05 kg | neck sustained 0.97 | 0.97 | 0.39 | 1.41 m/s² | 0.60 m/s | 0.18 rad | 0.53 | **fails**: animal walk |
| `60-N-T-10` | narrow stance, target turns >= 3 mm | 10.84 kg | ankle A sustained 0.77 | 0.77 | 0.55 | 1.24 m/s² | 0.48 m/s | 0.18 rad | 0.69 | all pass |
| `60-N-H-10` | narrow stance, target turns >= 3 mm | 10.93 kg | knee speed 0.77 | 0.77 | 0.54 | 1.24 m/s² | 0.48 m/s | 0.18 rad | 0.69 | all pass |
| `60-M-T-10` | 20 % headroom + Froude 0.05 sizing | 12.58 kg | ankle A speed 0.77 | 0.76 | 0.72 | 1.26 m/s² | 0.58 m/s | -0.02 rad | 0.73 | **fails**: clearance |
| `60-M-H-10` | 20 % headroom + Froude 0.05 sizing | 12.69 kg | ankle A speed 0.78 | 0.76 | 0.71 | 1.27 m/s² | 0.57 m/s | -0.04 rad | 0.70 | **fails**: clearance, turning targets |
| `60-N-T-10` | 20 % headroom + Froude 0.05 sizing | 12.54 kg | ankle A speed 0.77 | 0.77 | 0.62 | 1.27 m/s² | 0.58 m/s | -0.07 rad | 0.70 | **fails**: clearance, turning targets |
| `60-N-H-10` | 20 % headroom + Froude 0.05 sizing | 12.63 kg | ankle A speed 0.77 | 0.77 | 0.61 | 1.27 m/s² | 0.58 m/s | -0.13 rad | 0.69 | **fails**: clearance, turning targets |

Reading it (friction need with the stance foot's yaw moment from each body's turning screen; round-two body in
brackets; the governing drive is over each body's own sizing walks, which include the Froude 0.05 walk only on the
carpet-ready, targeted and feet bodies and on the blanket-headroom case):

* **Narrow stance, turns cleared by at least 3 mm** (`narrow_3mm`): the legs move out 6 mm per side instead of 4 mm;
  the turns then clear by 3.1–3.5 mm instead of 1.2–1.6 mm, at 274 mm overall width (narrow 270 mm, base 279 mm)
  and feet 0.36–0.37 H apart (base 0.42 H). It keeps the narrow stance's grip advantage: the nominal walk needs
  0.48–0.49 against 0.55–0.57 on the base bodies (−13 %), and the brisk walk 0.75–0.77 against 0.85–0.87. Its
  scorecard total rises to 0.69 (0.694 Tapered, 0.689 Hybrid; ranks 4 and 5), from 0.67 for the round-two narrow
  bodies; with the carpet-ready ankle drives added the two bodies rank first and third (§6, §7.6).
* **Longer toes** (×1.15): the walk's friction need falls by about 8 % (0.521 [0.565] at 60 cm Tapered nominal; 0.80
  [0.87] brisk), and the one-foot stop rises from 1.24 to 1.36 m/s². But the longer lever loads the upper ankle drive:
  0.88 of its sustained limit, against 0.76, which becomes the governing ratio in the target turns (0.87–0.89 against
  0.76–0.78) and lowers the natural-walking score.
* **Larger pads** (×1.25 in each direction): the friction need does not move (0.563 [0.565]): the stance foot's yaw
  moment acts over the pads' spread, which a larger pad barely changes. The one-foot stop rises to 1.33 m/s².
* **Both** (toes ×1.15 and pads ×1.25): friction −8 %, and the one-foot stop reaches 1.45–1.46 m/s² at 60 cm (1.41 at
  75 cm), close to the spec's 1.5 m/s², with the same ankle load as longer toes alone. §7.5 combines it with the
  refined bodies.
* **No calf fairing:** 0.13–0.14 kg lighter at 60 cm and 0.21 kg at 75 cm, every round-two gate passes (the 75 cm row
  keeps the round-two drives, so it fails the Froude 0.05 gate as round two did), and the friction need falls by
  about 2–3 %. At 60 cm the knee then stops at 0.80–0.86 rad, still short of the carpet swing; at 75 cm at 1.00 rad.
* **Blanket 20 % headroom at 60 cm** (`headroom_60`): RobStride 06 hip-pitch and knee drives, X4-36-L thigh-twist and ankle drives, and an X4-36-L hip roll on the
  base-width bodies (the narrow ones keep the AK45-36 hip roll),
  12.5–12.7 kg (+1.7–1.8 kg) and 35–36.5 min (−10–11 min), for a governing ratio of 0.77–0.78. The larger drives no
  longer fit: the thigh meets the calf fairing in the brisk walk at the search's limit of 0.50 for the ankle drives,
  so every body fails clearance. Not pursued.


### 7.5 Bigger feet on the refined bodies

Longer toes (×1.15) and larger pads (×1.25) combined with the carpet-ready 60 cm and targeted 75 cm bodies
(`round2/refine/feet_60`, `feet_75`), re-sized and screened:

| Body | Mass | Runtime | Friction need: walk / turn / brisk | One-foot stop | Medium pile, servos held | Target turns (governing) | Froude 0.05 walk |
|---|---|---|---|---|---|---|---|
| `60-M-T-10` carpet-ready + feet | 10.92 kg (10.83) | 45.6 min (46.5) | 0.51 / 0.58 / 0.78 (0.55 / 0.63 / 0.86) | 1.46 m/s² (1.25) | stands (tips) | 0.87, ankle A (0.76) | knee 0.98 (0.95) |
| `60-M-H-10` carpet-ready + feet | 11.02 kg (10.93) | 45.3 min (46.3) | 0.50 / 0.57 / 0.76 (0.54 / 0.62 / 0.83) | 1.46 m/s² (1.25) | stands (tips) | 0.87, ankle A (0.77) | knee 0.96 (0.97) |
| `75-M-T-10` targeted + feet | 14.71 kg (14.58) | 37.3 min (37.6) | 0.37 / 0.42 / 0.57 (0.40 / 0.46 / 0.63) | 1.41 m/s² (1.20) | stands (stands) | 1.00, thigh twist (0.78) | thigh twist 0.93 (knee 0.95) |
| `75-M-H-10` targeted + feet | 14.85 kg (14.71) | 36.7 min (37.3) | 0.36 / 0.41 / 0.56 (0.39 / 0.45 / 0.61) | 1.41 m/s² (1.20) | stands (tips) | 0.77, thigh twist (0.78) | knee 0.93 (0.97) |

The reference body's value is in brackets. Bigger feet cut the friction need by about 8 %, raise the one-foot stop
by about 17 % (to within 3–6 % of the spec's 1.5 m/s²) and let every one of these bodies stand on medium pile with
its servos held, which no round-two 60 cm body could. The cost is the upper ankle drive: at 60 cm it rises from
0.76 to 0.87 of its sustained limit and becomes the governing drive in the target turns. At 75 cm the re-sizing
spends the margin differently: the Tapered body falls back to the lighter CubeMars AK45-36 thigh twist, which then
sits at its limit in the turns (1.00), so it would need the X4-36-L kept; the Hybrid keeps its drives. On the scorecard (§6) the 75 cm Hybrid with bigger feet totals 0.702, the best 75 cm body and second of all (0.664 without the feet); the Tapered one falls to 0.551 (rank 35), its thigh twist at the limit. At 60 cm the bigger feet lower the totals to 0.63–0.64 (from 0.66–0.68): the upper ankle drive's higher load costs more walking and turning score than the harder stop adds to robustness.

### 7.6 Narrow carpet-ready 60 cm (picks 1 and 3)

The narrow stance with every turn clearing by at least 3 mm (`narrow_3mm`, §7.4) and the carpet-ready ankle
drives (§7.1) together (`round2/refine/narrow_carpet_60`): the narrow layout with the legs moved out 6 mm per
side (round two's narrow bodies: 4 mm), the ankle drives at 0.60 of the shank, and the body sized with the Froude
0.05 walk and the nominal walk lifted to 0.19 H as extra sizing walks. Both bodies keep every drive of the base-width carpet-ready
bodies:

| Body | Mass | Runtime | Cost nominal / high | Governing limit | Target turns (governing) | Turn gaps: target turns; 0.5 rad/s walk | Friction need: walk / turn / brisk | Tightest joint margin | Score (v2) |
|---|---|---|---|---|---|---|---|---|---|
| `60-N-T-10` narrow carpet-ready | 10.81 kg (10.83) | 47.6 min (46.5) | $14.8k / $20.5k | knee speed 0.93 (0.95) | 0.76, neck servo (0.76, thigh twist) | 3.4; 3.1 mm (4.0; 4.0) | 0.48 / 0.54 / 0.75 (0.55 / 0.63 / 0.86) | 0.18 rad (0.20) | 0.703 (0.675) |
| `60-N-H-10` narrow carpet-ready | 10.91 kg (10.93) | 47.4 min (46.3) | $14.8k / $20.6k | knee speed 0.95 (0.97) | 0.76, ankle A (0.77, thigh twist) | 3.4; 3.0 mm (4.0; 3.4) | 0.47 / 0.53 / 0.73 (0.54 / 0.62 / 0.83) | 0.18 rad (0.15) | 0.699 (0.658) |

The base-width carpet-ready body's value is in brackets (`carpet_60`, §7.1). Against it the narrow stance needs
12–14 % less friction in every walk and turn, draws 1.5–1.6 W less walking (69.3–69.6 W), so runs about 1 min
longer, and brings the knee's speed in the Froude 0.05 walk down from 0.95–0.97 to 0.93–0.95. Its risk is the
turn gaps: 3.4 mm in the target turns against 4.0 mm at base width (3.0–3.1 mm against 3.4–4.0 mm in the
0.5 rad/s walking turn, a stretch case). The one-foot stop (1.25 m/s²), the
push (0.48–0.49 m/s), the camera and the person-in-view check are unchanged.

The carpet swing, with both reference walks lifted (`carpet_lift_narrow_carpet_60_0.190_0.195.json`) and with the
nominal walk alone lifted, the brisk walk at its reference lift (`carpet_lift_narrow_carpet_60_0.190_0.195_nominal_only.json`):

| Lift | `60-N-T-10`: above a high-pile top (target 24.3 mm) | `60-N-H-10` (target 24.1 mm) | Both walks lifted: drives (knee); walking gates | Nominal walk alone: drives (knee); walking gates |
|---|---|---|---|---|
| 0.19 H | 24.3 mm: met | 24.0 mm: 0.05 mm short | over (1.05 / 1.06, brisk walk); pass | within (0.84 / 0.85); pass |
| 0.195 H | 25.4 mm: met | 25.0 mm: met | over (1.06 / 1.08, brisk walk); Tapered pass, Hybrid knee margin 0.097 rad, under the 0.10 rad gate | within (0.85 / 0.86); pass (Hybrid knee margin 0.13 rad) |

So on carpet the narrow Tapered body walks at the nominal speed with a 0.19 H swing and the narrow Hybrid with a
0.195 H swing, both within their drives and every walking gate; a brisk walk with the carpet swing exceeds the
knee's speed on both, as at base width. The Hybrid's knee-range margin falls under the gate only if the brisk walk
is lifted too, which its knee's speed already rules out.

## 8. Size study: a smaller 60 cm, larger 75 cm motors, or 80 cm? (F19)

You asked whether a 60 cm robot 5–10 % smaller would gain margin, and whether the 75 cm robot could take larger
motors or would need 80 cm. `designlab/sizestudy.py` sized and screened each case like the round-two candidates
(`round2/size_study/SIZE_STUDY.md`). Study heights 54 and 57 cm use the 60 cm class's packaging rules; 80 cm uses
the 75 cm class's.

| Case | Mass | Governing drive | Froude 0.05 walk | Runtime | Cost nominal | Gates |
|---|---|---|---|---|---|---|
| 60 cm, round two (`60-M-T-10` / `60-M-H-10`) | 10.85 / 10.96 kg | 0.77 / 0.79 | within (knee 0.96 / 0.98) | 46.4 / 46.2 min | $14.8k / $14.9k | pass |
| 57 cm (5 % smaller), drives re-selected | 10.58 / 10.68 kg | 0.78 / 0.80 | within (knee 0.98) | 48.2 / 47.7 min | $14.6k | **fail**: clearance |
| 54 cm (10 % smaller), drives re-selected | 10.29 / 10.39 kg | neck 0.96–0.97 | **over** (knee 1.02–1.04) | 50.1 / 49.6 min | $14.3k | **fail**: clearance, Froude 0.05 |
| 75 cm, targeted upgrade (§4) | 14.58 / 14.71 kg | knee speed 0.95 / 0.97 | within | 37.6 / 37.3 min | $18.4k | pass |
| 75 cm, 20 % headroom on every drive + Froude 0.05 | 19.66 / 20.00 kg | 0.73 / 0.75 | within | 61.9 / 68.9 min | $23.0k / $20.3k | **fail**: no battery path |
| 80 cm, round-two sizing | 15.64 / 15.81 kg | 1.00 / 0.97 | **over** (hip pitch 1.03–1.04) | 36.2 / 35.6 min | $18.0k / $18.3k | **fail**: Froude 0.05 |
| 80 cm, 20 % headroom + Froude 0.05 | 22.15 / 22.25 kg | 0.80 | within | 64.2 min | $21.1k / $21.2k | **fail**: no battery path |

**A smaller 60 cm robot does not gain margin.** Shrinking by 5 % saves only 0.27–0.28 kg, and by 10 % only
0.56–0.57 kg, because about half the mass is catalog hardware that does not shrink: the drives, battery,
compute, power and control boards. Two things then get worse. First, the parts that do not shrink stop fitting:
at 57 and 54 cm the hip-roll motor overlaps the power board (by 1.6–22.7 mm) and the two ToF sensors overlap
(0.6–2.0 mm), which fails the clearance gate. A hand re-layout might recover the packaging, but not more than the
0.3–0.6 kg. Second, the drives run faster: for the same Froude number a smaller robot steps faster
(stride time scales with the square root of its size), so at 54 cm the knee exceeds its speed limit in the
Froude 0.05 walk (1.02–1.04). The highest torque ratio, the thigh twist's, barely moves (0.77–0.78 at 60 cm,
0.75–0.79 at 54 cm), and neither does the hip roll's (0.69 → 0.66–0.67). The hip pitch, ankles and neck gain torque
margin (at 54 cm with the 60 cm drives 0.33–0.36, 0.65 and 0.58, against 0.54–0.59, 0.76 and 0.76), but the thigh
twist still sets the torque limit and the knee's speed gets worse. Keeping the 60 cm drives on the smaller bodies
gives the same result.

**Larger motors at 75 cm: yes, targeted; no need for 80 cm.** Changing four drives (§4) costs $1.0k nominal,
takes 0.32 kg off, passes every gate and lifts the 75 cm body's scorecard total from 0.52 to 0.67, level with the
best of the 20 round-two 60 cm candidates (0.67) but below the refined narrow carpet-ready 60 cm picks (0.70) (§6).
Requiring 20 % headroom on every drive (with the Froude 0.05 walk as a sizing case) instead snowballs (it changes six
of the eight drive groups on the Tapered body and five on the Hybrid): the bodies grow to about 20 kg, need a 12S pack that has no
removal path through the tail-base cassette, and fail clearance (the Hybrid: 20.0 kg on 48 V, +$2.8k). At 80 cm
the round-two sizing hits the same limits as 75 cm (the Tapered thigh twist at 1.00, the hip pitch over in the
Froude 0.05 walk at 1.03–1.04) and is 0.7–0.8 kg heavier; with headroom it reaches 22 kg. Nothing at 80 cm is
better than the targeted 75 cm body. (The `75-M-T-10` headroom record, 19.7 kg and +$5.6k, predates the
bus-oscillation fix F17 and pairs a 12S pack with the 24 V bus, so it is indicative only.)

Two analysts wrote predictions blind to these runs (`round2/size_study/PREDICTIONS.json`; written while the runs
were in progress, without opening their outputs). They matched the 57 and
54 cm masses within 0.05 kg (57 cm within 0.03 kg) and the targeted 75 cm mass, cost and runtime within 0.01 kg,
$10 and 0.1 min, though the latter were computed as part swaps on the simulated what-if record. They
underestimated the blanket-headroom 75 cm mass by 0.9 kg on the Hybrid.

## 9. Materials and power (F22)

Full note with sources: `round2/refine/MATERIALS.md`. Each lever was applied to the carpet-ready `60-M-T-10` and
the targeted `75-M-T-10`, the leading bodies when these runs were made, with the reference body's drives and pack
kept (the margin view below, `round2/refine/fixed_drives.json`). Four were also re-sized: the 2 mm carbon plates, the
PA12 walls ×0.75, the 2 mm carbon + PA12 combination and tether-only power (`round2/refine/mat60_cf`, `_thin`,
`_light`, `_tether` and the `mat75_` equivalents). The narrow 60 cm picks keep the
same drives at nearly the same mass; the 75 cm pick is the Hybrid with bigger feet. The levers were not re-run on
the picks, so expect similar, not identical, numbers.

| Lever (same drives) | 60 cm: mass / runtime / cost nominal | 75 cm: mass / runtime / cost nominal | Neck servo (60; 75) | Knee speed |
|---|---|---|---|---|
| Reference | 10.83 kg / 46.5 min / $14.8k | 14.58 kg / 37.6 min / $18.4k | 0.76; 0.67 | 0.95; 0.95 |
| Carbon-fibre frame plates, sourced 3 mm quasi-isotropic | −0.39 kg / +1.1 min / +$1.6k | −0.44 kg / +1.0 min / +$2.1k | 0.75; 0.66 | 0.95; 0.95 |
| Carbon plates at equal rigidity, next stock gauge (3.99 mm; 3.43 mm needed) | −0.20 kg / +0.5 min / +$1.6k | −0.22 kg / +0.4 min / +$2.1k | 0.76; 0.66 | 0.95; 0.95 |
| 6061 plates at 2.54 mm (brackets 3 mm) | −0.17 kg / +0.5 min / $0 | −0.16 kg / +0.3 min / $0 | 0.76; 0.66 | 0.95; 0.95 |
| Covering walls ×0.75, PA12 | −0.35 kg / +1.4 min / −$0.3k | −0.59 kg / +1.7 min / −$0.4k | 0.69; 0.59 | 0.95; 0.94 |
| **Covering walls ×0.75, PA11 (recommended)** | −0.31 kg / +1.2 min / +$20 | −0.52 kg / +1.5 min / +$30 | 0.70; 0.60 | 0.95; 0.94 |
| Carbon plates + PA11 walls ×0.75 | −0.70 kg / +2.3 min / +$1.6k | −0.95 kg / +2.5 min / +$2.1k | 0.69; 0.60 | 0.95; 0.94 |
| Tether-only (no pack or cassette) | −0.61 kg / tethered / −$0.1k | −0.59 kg / tethered / −$0.1k | 0.76; 0.67 | **1.07; 1.03** |

* **Lighter materials do not move the binding limit.** The knee's speed in the Froude 0.05 walk is set by the
  walk's geometry and timing, not by mass: every material case leaves it at 0.94–0.95. They buy torque headroom
  and runtime, and runtime is already about twice the 15–20 min requirement. Re-sized instead, the 60 cm body
  keeps every drive and its pack and simply comes out lighter (its torque ratios fall, e.g. thigh twist 0.765 →
  0.69–0.74; the knee speed stays 0.95). The 75 cm loop spends the saving on a lighter thigh twist (back to the
  AK45-36, at 0.91–0.98 of its limit; with 2 mm carbon and thin walls together, also the W150 neck), so its
  governing ratio stays high (0.93–0.98).
* **Structure: keep 6061-T6 for prototype 1.** 3 mm carbon plate saves 0.39–0.44 kg for $1.6–2.1k (a model
  estimate, ±50 % until quoted) and takes the 75 cm body to $20.5k nominal, over P4's $20k; but it has only 0.67×
  the rigidity of 3 mm aluminium, and at equal rigidity, using the 3.99 mm stock gauge (3.43 mm is needed but not stocked), it saves 0.20–0.22 kg.
  Thinner 6061 plate (2.54 mm, 0.61× the rigidity) saves 0.16–0.17 kg at no extra cost. Carbon cannot hold bearing fits or threads, needs
  bonded aluminium inserts, delaminates out of sight in a fall, and the quasi-isotropic plate is not a stock 2–4 day
  part. 7075-T6 suits the hip and thigh brackets if falls bend them; titanium only for pins and fasteners in carbon
  joints; magnesium is not recommended. A hybrid frame (carbon plates and tube spars, aluminium wherever a motor,
  bearing or thread attaches) is a direction for the 3–4 ft robot, whose long members (the spine, and at 120 cm the tibias) are
  stiffness-governed in the model, although most of its plates still sit at the 3 mm minimum.
* **Coverings: MJF PA11 with graded walls (1.4 / 1.2 / 1.0 mm by panel size).** PA11 has 2.5× the elongation and
  1.4× the notched impact strength of PA12, so thin walls dent instead of cracking under knocks and falls (36 J at
  60 cm, 63 J at 75 cm). It saves 0.31 / 0.52 kg at about today's cost (about 0.26 / 0.44 kg if the thinner walls
  need more internal supports), mostly off the trunk and legs (the head and tail panels are 28–29 % of it),
  although the largest relief is on the neck and tail servos, where that mass sits on a lever arm.
  Add EVA foam bumpers at strike points (snout, jaw tip, tail tip, shins) and stretch fabric only over moving
  gaps. Moulded carbon skins (0.44–0.56 mm) are the lightest rigid shell but need moulds, 3–8 weeks and an
  estimated $2.4–14.2k.
* **Foot pads: cast polyether polyurethane, 50–60 Shore A, on a 70–80 A core, with 1.0–1.5 mm round studs.**
  Expected static friction (estimates from the sources): carpet 0.70–0.85, wood 0.8–0.9, ceramic tile 0.9–1.0,
  polished concrete 0.6–0.85 dry, laminate and vinyl 0.42–0.8 (as low as about 0.50 on vinyl and 0.42 on
  laminate). With the stance foot's yaw moment the narrow carpet-ready 60 cm picks need 0.47–0.48 walking,
  0.53–0.54 turning and 0.73–0.75 brisk; the 75 cm pick with bigger feet 0.36, 0.41 and 0.56. Taking vinyl and
  laminate at those low values, the 75 cm pick walks and turns on every floor class (on laminate its
  turns have almost no margin) and walks brisk on all but vinyl and laminate; the 60 cm picks walk and turn on
  carpet, wood, tile and polished concrete, walk brisk on wood and tile but not on the lowest-grip carpet or
  concrete, barely walk on vinyl and cannot walk on laminate. The base-width 60 cm bodies needed 0.54–0.55, 0.62–0.63
  and 0.83–0.86, too much for vinyl at any pace. Measure the chosen pad at the robot's pad pressure (72–74 kPa
  at 60 cm, 41 kPa at 75 cm with the larger pads) on the owner's floors before relying on any of these (about a day
  and $100, estimate).
* **Power: battery plus an external DC input; tether-only is not recommended.** The pack is also the tail-base
  counterweight (F2). Without it the centre of mass moves 12–16 mm forward, the hip-pitch hold rises 14–37 % and,
  with the same drives, the knee goes past its speed limit. The recommended design keeps the pack on every run
  and adds a 24 V input through a priority power path (it switches between sources, so the tether also carries the
  peaks, 6.2 / 8.7 A), a clamp at 27 V or below for the 60 cm AK45 drives, a Category 1 E-stop that cuts both
  sources, a tether hung from a boom (never trailed: 5 m of floor cable drags 1.2–3.6 N against a 7–12 N tip
  threshold at the cable exit) and an equal-mass dummy pack for battery-free runs. A 6S1P Li-ion pack (Molicel
  P45B, 434 g, 97–100 Wh) would add about 15–33 % runtime for 72 g less, if the cassette is redesigned.
* **The 3–4 ft robot: battery, yes.** With the Pi 5's budget the round-one 100 and 120 cm Tapered and Hybrid base bodies draw about 130–190
  W walking; a 0.86–1.72 kg Li-ion pack gives about 50–94 min, 2–4 % of their mass. The research recommends 48 V
  and two hot-swappable strings of sub-100 Wh modules (they fly as carry-on spares), a protected cassette inside
  the frame (a fall carries about 160–300 J at that size), and the tether only on a fall-arrest gantry and a
  charging dock.

## 10. Head and tail movement

The same on every body (every MuJoCo model under `round2/`: `round2/models/*.xml`, `round2/refine/*/models/*.xml`,
`round2/size_study/*/models/*.xml`, `round2/whatif_fr05/models/*.xml`; `designlab/gait.py`):

* **Head:** one nodding joint at the base of the neck, ±0.35 rad (±20°). The S-curved neck and the skull pitch
  together. The head does not turn side to side, the jaw is not powered yet (deferred) and the arms are fixed.
* **Tail:** one side-to-side joint at the tail base, ±0.45 rad (±26°). The tail is one rigid spar; the Hybrid's
  segments are covers over it.
* **In the reference walk:** the head nods ±0.05 rad (±3°) at the step frequency and the tail sways ±0.12 rad (±7°)
  at the stride frequency, inside the behavior spec's ±0.05–0.15 rad and ±0.1–0.2 rad (`BEHAVIOR_SPEC.md` §1).
* **Servos on the recommended bodies:** at 60 cm the Feetech STS3250 neck servo runs at 0.76 of its sustained
  rating and the tail servo at 0.43–0.48 of its peak rating; at 75 cm the targeted XM540-W270 neck runs at 0.66 and
  the tail servo at 0.79 of its peak rating (0.89 on the targeted Tapered body without the bigger feet), both in the
  animal-like walk. There is room for livelier head motion, but less for a faster tail at 75 cm. On the bodies the
  material runs used, thinner PA11 coverings cut the neck to 0.70 (60 cm) and 0.60 (75 cm) and the 75 cm Tapered
  tail peak to 0.77 (§9).
* **Following limits the head:** the camera is in the head; a person stays in view only with the head near level.

## 11. Corrections to round one

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
    an ankle drive on 13 (all 60 cm), the tail drive on 10, the knee on 2 and the hip pitch on 1. The holds and
    continuous loads behind round one's size and family comparisons are unchanged.
* **Turning in place needs several times the thigh-twist speed first stated.** `SCREENING.md` §7 said a 0.5
  rad/s turn in place needs about 0.5 rad/s. It needs about 2.4–2.6 rad/s before the margin on the four 10°
  round-one bodies screened. The 18 round-one bodies with a CubeMars AK80-64 twist drive on 24 V (100 and 120 cm)
  would miss the 1.15 speed margin (`100-W-T-10`: 0.97 of the usable speed without it). Corrected in
  `SCREENING.md` §7.
* **Round two's own documents** were audited overnight (681 checkable claims against the records, each reported
  problem checked by two independent skeptics). The corrections are in this review, `DECISION_LOG.md` (F9–F16,
  A13, A14), `SCREENING.md`, `COMPUTE.md` and `README.md`. The largest: friction needs that include the stance
  foot's yaw moment are much higher than the carpet gate's force ratio, so grip, not turning, is the 60 cm weak
  spot (§2, F21); and the 75 cm carpet friction range is 0.21–0.24, not 0.22–0.24. Two later audits checked the
  overnight documents (406 and 2,232 claims); their confirmed corrections are applied, and none changes a pick
  (`round2/AUDIT.md`, `round2/audit_final.json`).
* `ROUND1_REVIEW.md`, `SCREENING.md` §3 and §7 carry the round-one corrections; round one's generated tables are
  not regenerated.

## 12. Questions for the owner

1. **Bodies for verification.** The recommended set (Summary, §6) is the scorecard's three picks. Accept it, swap
   one (for example the targeted 75 cm Tapered for its looks, or the base-width carpet-ready 60 cm Tapered for its
   wider turn gaps), or name a different set.
2. **Priority weights.** Natural walking and turning first, then robustness, following, carpet, endurance, head
   and tail, packaging and cost. Keep them, or reorder?
3. **75 cm drives.** (a) The targeted upgrade (recommended: −0.32 / −0.33 kg, +$1.0k, −0.6 / −0.7 min against round two,
   Tapered / Hybrid); (b) the
   targeted upgrade plus a faster RobStride 06 knee (knee speed 0.95–0.97 → 0.55–0.57 of its limit; +0.84–0.85 kg, −6 min, friction need in the turn 0.45–0.46 → 0.49–0.51; the scorecard puts it level with (a) on the Tapered body and below the recommended Hybrid with bigger feet, 0.68 against 0.70); (c) keep the round-two sizing.
4. **Calf fairings.** (a) Keep them, with the 60 cm ankle drives lowered to 0.60 of the shank (recommended: the
   carpet-ready bodies); (b) drop them (exposed ankle drives and push rods: lighter by 0.13–0.14 kg at 60 cm and 0.21 kg at 75 cm, easier to service, more robotic; at 60 cm the knee then stops at 0.80–0.86 rad, so the ankle drives must still move down for the carpet swing).
5. **Feet.** The recommended 75 cm body has longer toes (×1.15) and larger pads (×1.25): about 8 % less grip
   needed, a one-foot stop of 1.41 m/s² instead of 1.20, and standing on medium pile with the servos held, for
   0.15 kg. At 60 cm the same feet load the upper ankle drive to 0.87 of its limit (0.76) and lower the scorecard
   total, so the 60 cm picks keep the round-two feet (§7.5). (a) As recommended; (b) the bigger feet on every body;
   (c) the round-two feet on every body.
6. **Materials and power.** (a) The recommended set: 6061-T6 frame (2.54 mm plates where a leg bench test allows),
   MJF PA11 coverings with walls about a quarter thinner, cast polyurethane pad treads measured on your floors, and
   the battery plus an external DC input; (b) change any of these (say which); carbon-fibre plates and a tether-only
   body are not recommended (§9).
7. **Head camera.** Keep it in the skull roof (person in view; the near floor relies on the ToF pair), add a low
   forward camera for the near floor (cost, power and wiring), or move the camera forward and lower (better floor
   view, less of a close person's torso)?
8. **Stage 1 go-ahead.** May the bounded closed-loop probe start on the chosen bodies (§13)?

The answers go into `DECISION_LOG.md` as P-entries.

## 13. Next

* **Stage 1, closed-loop probe** (needs the owner's go-ahead): a bounded controller study on the chosen
  bodies, with the same budget each. It covers walking at the target speeds and the animal-like pace, the target
  turns, start and stop, a push, and walking with the carpet swing on the carpet floors of this round. It is not
  the final training; it checks that a controller can use the margins found here.
* **Bench tests that need no robot:** friction and twist tests of the chosen pad compound at the robot's pad
  pressure on the owner's floors, dry and dusty (about a day and $100, estimate), and a measurement of the Pi 5
  set's draw at the converter input with the detector, camera and logging running (A14).
* **Stage 2, single-leg bench:** one leg of the leading body. Drive temperatures over a walking duty cycle, the
  knee and ankle clearances with the ankle drives at their new station, and the knee's speed in the animal-like
  walk, which governs all three picks.
* **Verification training:** equal-budget training on the finalists against `BEHAVIOR_SPEC.md`, then one design
  is chosen.

## 14. What this round does not show

* No controller has walked, turned or stood on carpet with any of these models. The turns, walks and stance
  checks are prescribed motions and simple balance models.
* Slip, impacts, touchdown dynamics and carpet damping are not modelled (assumption A16); thermal behaviour is
  the sustained planning fraction only (A1). Pad friction is the carpet band's lower edge until measured.
* Material properties are sourced; the carbon allowable, the PA11 price and the covering and pad estimates are
  marked as estimates in `round2/refine/MATERIALS.md`. The RobStride 06's winding resistance is not published,
  so the faster-knee bodies' power and runtime use the model's fallback value.
* All masses, powers, runtimes and costs are model estimates from sourced part data and stated rules. None is
  measured.
