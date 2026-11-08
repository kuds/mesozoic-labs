# Materials and power (round two, overnight)

Owner questions (overnight, 7–8 Nov): which materials for the structure and the cosmetic coverings, carbon fibre
or other light, strong materials; anything on the margin worth adjusting; a non-battery (direct power) version;
battery operation on the 3–4 ft robot.

Method. Sourced research on the materials, coverings, foot pads and power architecture (cited inline; accessed
2026-11-08; the research workflow's raw output is in the session scratch). Simulation of each lever on the two
refined leaders, `60-M-T-10` carpet-ready (`round2/refine/carpet_60`) and `75-M-T-10` with the targeted upgrade
(`round2/refine/targeted_75`) (`designlab/refine.py`). They led when these runs were made. The final picks
(ROUND2_REVIEW §6) are the narrow carpet-ready 60 cm bodies, which keep the same drives at nearly the same mass
(10.81–10.91 kg against 10.83–10.93 kg), and the targeted 75 cm Hybrid with bigger feet (14.85 kg); the levers were not
re-run on them, so expect similar but not identical numbers.

* **same drives**, every lever (`round2/refine/fixed_drives.json`, `REFINE.md` last table): the lighter body keeps
  the reference body's drives and pack, so the saving shows up as margin;
* **re-sized**, for the 2 mm carbon plates, the PA12 walls ×0.75, their combination and tether-only power
  (`round2/refine/mat60_cf`, `_thin`, `_light`, `_tether` and the `mat75_` equivalents): the normal sizing loop picks
  drives and pack for the lighter body.

Every number is a model estimate. None is measured.

## Short answers

1. **Lighter materials do not move the limit that binds the leading bodies.** On both refined leaders the
   governing limit is the knee's *speed* in the Froude 0.05 walk (0.95). That is set by the walk's geometry and
   timing, not by mass. Every material case leaves it at 0.94–0.95 with the reference drives (0.93–0.95 re-sized). Lighter bodies lower the torques, the
   walking power and the neck and tail loads, so they buy torque headroom and runtime, not walking-speed
   headroom. Runtime already exceeds the 15–20 min requirement by about 2×.
2. **Carbon-fibre frame plates: not for prototype 1.** With the sourced 3 mm quasi-isotropic plate they save
   0.39 kg (60 cm) or 0.44 kg (75 cm) for $1.6k or $2.1k more (nominal, a model estimate to treat as ±50 % until
   quoted), and they take the 75 cm body to $20.5k nominal, over P4's $20k. A 3 mm carbon plate has only 0.67× the
   rigidity of today's 3 mm aluminium; at equal rigidity (3.99 mm stock) the saving shrinks to 0.20 kg (60 cm) and 0.22 kg (75 cm) for the same cost. Thinner
   aluminium plate (2.54 mm, also stocked) gets 0.18 kg (60 cm) and 0.16 kg (75 cm) at no extra cost. Carbon cannot hold bearing fits or
   threads, delaminates out of sight in a fall, needs bonded aluminium inserts, and the quasi-isotropic plate is not
   a stock 2–4 day part. Keep 6061-T6. A hybrid frame (carbon plates and tube spars, aluminium wherever a motor,
   bearing or thread attaches) is a direction for the long-term robot, whose long members (the spine, and at 120 cm
   the tibias) are stiffness-governed in the model; most of its plates still sit at the 3 mm minimum.
3. **Coverings: the best value lever.** Walls about a quarter thinner save 0.31–0.35 kg (60 cm) or 0.52–0.59 kg
   (75 cm) at about today's cost (PA12 −$0.3k to −$0.7k, PA11 +$20 to +$60). Most of the mass comes off the trunk
   and legs (the head and tail panels are 28–29 % of it), but the largest relief is on the neck and tail servos,
   where mass sits on a lever arm. The research
   recommends MJF **PA11** for the thinner walls instead of today's PA12: 2.5× the elongation and 1.4× the notched
   impact strength, so walls dent rather than crack, for about 30 % more per volume, which the thinner walls pay
   for. With PA11 at ×0.75 the neck servo falls from 0.76 to 0.70 of its rating (60 cm) and from 0.67 to 0.60
   (75 cm), and the 75 cm tail servo's peak from 0.89 to 0.77; cost changes by $20–60. If the thinner walls need
   more internal supports (25 % and 29 % instead of 20 % and 24 %), the saving is about 0.26 / 0.44 kg.
4. **Foot pads: grip is the limit after the drives.** Counting the stance foot's yaw moment (F21), the narrow
   carpet-ready 60 cm picks need friction 0.47–0.48 walking, 0.53–0.55 in the 0.35 rad/s turn and 0.73–0.75 brisk
   (the base-width carpet-ready bodies 0.54–0.55, 0.62–0.63 and 0.83–0.86); the targeted 75 cm Hybrid with bigger feet
   needs 0.36, 0.41 and 0.56 (the targeted bodies with round-two feet 0.39–0.40, 0.45–0.46 and 0.61–0.63). A cast
   polyether polyurethane tread (50–60 Shore A, with 1.0–1.5 mm studs) is expected to give about 0.70–0.85 on carpet
   and 0.8–1.0 on wood and tile, but laminate and vinyl can be as low as 0.42–0.50. So the 60 cm robot may barely
   walk on vinyl and not at all on laminate, and both sizes need speed and turn limits for brisk walking on slippery
   floors. Measure the chosen pad on the owner's floors before relying on any of this.
5. **Direct power (tether-only): no.** The battery is also the counterweight at the tail base (F2). Without it the
   centre of mass moves 12–16 mm forward, the hip-pitch hold rises 14–37 %, and with the same drives the knee goes
   past its speed limit (1.03–1.07). Keep the removable battery and add an external DC input, as P4 already asks
   ("removable battery … external-power operation"); use the tether for bench work, hung from a boom.
6. **3–4 ft robot: battery operation is easy in mass terms.** At 100–120 cm a 0.9–1.7 kg pack gives about 50–94
   min of walking and is 2–4 % of the robot's mass (round-one sizing with the Pi 5's power budget). The constraints there are pack safety in falls of a 28–56 kg robot,
   transport rules for packs over 100 Wh, and charging, not mass.

## 1. Where the mass is and what limits the bodies

| | `60-M-T-10` carpet-ready | `75-M-T-10` targeted |
|---|---|---|
| Total | 10.83 kg | 14.58 kg |
| Leg drives | 4.16 kg (38 %) | 5.24 kg (36 %) |
| Structure (twin 6061 plates, brackets) | 1.67 kg (15 %) | 2.40 kg (16 %) |
| Coverings + internal supports | 1.10 + 0.22 kg (12 %) | 1.77 + 0.43 kg (15 %) |
| Battery and cassette | 0.68 kg | 0.68 kg |
| Integration reserve (6 %) | 0.65 kg | 0.88 kg |
| Everything else (bearings, linkage, power, compute, harness, feet, sensors, control, fasteners, servos) | 2.35 kg | 3.2 kg |
| Governing limit | knee speed 0.95 (Froude 0.05 walk) | knee speed 0.95 (Froude 0.05 walk) |
| Highest torque ratio (leg drives) | thigh twist, sustained 0.76 | thigh twist, sustained 0.78 |

Only the structure, the coverings and the 6061 spine beam and neck spar (booked under linkage) change with material: about 30 % and 35 % of the mass, and the 6 % integration reserve follows the total. Most 60 cm plates sit at the 3 mm minimum
thickness, so their mass is density × minimum thickness; at 75 cm the tibia and spine are stiffness-governed.

## 2. Simulated levers

Same drives as the reference body (the margin view). Δ is against the reference.

| Lever | 60 cm Δ mass | 60 cm: walking power / runtime | 60 cm cost Δ (nominal / high) | 75 cm Δ mass | 75 cm: walking power / runtime | 75 cm cost Δ | Knee speed (60 / 75) | Highest leg torque (60 / 75) | Neck sustained / tail peak (60; 75) |
|---|---|---|---|---|---|---|---|---|---|
| Reference | 10.83 kg | 70.8 W / 46.5 min | $14.8k / $20.6k | 14.58 kg | 87.7 W / 37.6 min | $18.4k / $25.4k | 0.95 / 0.95 | 0.76 / 0.78 | 0.76 / 0.52; 0.67 / 0.89 |
| Carbon plates, sourced 3 mm quasi-isotropic | −0.39 kg | 69.2 W / 47.6 min | +$1.6k / +$2.3k | −0.44 kg | 85.5 W / 38.6 min | +$2.1k / +$3.1k | 0.95 / 0.95 | 0.74 / 0.75 | 0.75 / 0.51; 0.66 / 0.89 |
| Carbon plates at equal rigidity (3.99 mm quasi-isotropic) | −0.20 kg | 70.1 W / 47.0 min | +$1.6k / +$2.3k | −0.22 kg | 86.7 W / 38.0 min | +$2.1k / +$3.1k | 0.95 / 0.95 | 0.76 / 0.77 | 0.76 / 0.52; 0.66 / 0.90 |
| **6061 plates at 2.54 mm (brackets stay 3 mm)** | −0.18 kg | 70.1 W / 47.0 min | $0 / $0 | −0.16 kg | 86.9 W / 37.9 min | $0 / $0 | 0.95 / 0.95 | 0.75 / 0.77 | 0.76 / 0.52; 0.66 / 0.89 |
| Carbon plates, 2 mm floor (optimistic) | −0.62 kg | 68.3 W / 48.2 min | +$1.6k / +$2.3k | −0.81 kg | 83.6 W / 39.4 min | +$2.1k / +$3.1k | 0.95 / 0.95 | 0.72 / 0.73 | 0.75 / 0.51; 0.65 / 0.89 |
| Covering walls ×0.75 (PA12) | −0.35 kg | 68.8 W / 47.9 min | −$0.3k / −$0.5k | −0.59 kg | 84.0 W / 39.3 min | −$0.4k / −$0.7k | 0.95 / 0.94 | 0.74 / 0.75 | 0.69 / 0.44; 0.59 / 0.76 |
| **PA11 coverings, walls ×0.75 (recommended)** | −0.31 kg | 69.0 W / 47.7 min | +$20 / +$40 | −0.52 kg | 84.4 W / 39.1 min | +$30 / +$60 | 0.95 / 0.94 | 0.74 / 0.75 | 0.70 / 0.45; 0.60 / 0.77 |
| Sourced carbon + walls ×0.75 | −0.74 kg | 67.3 W / 49.0 min | +$1.3k / +$1.8k | −1.02 kg | 81.9 W / 40.3 min | +$1.7k / +$2.4k | 0.95 / 0.94 | 0.71 / 0.72 | 0.68 / 0.43; 0.59 / 0.75 |
| Sourced carbon + PA11 walls ×0.75 | −0.70 kg | 67.5 W / 48.8 min | +$1.6k / +$2.3k | −0.95 kg | 82.3 W / 40.1 min | +$2.1k / +$3.2k | 0.95 / 0.94 | 0.71 / 0.73 | 0.69 / 0.44; 0.60 / 0.77 |
| Tether-only (no pack or cassette; 0.12 kg connector) | −0.61 kg | 70.4 W / tethered | −$0.1k / −$0.1k | −0.59 kg | 87.9 W / tethered | −$0.1k / −$0.0k | **1.07 / 1.03** | 0.72 (thigh twist) / 0.82 (hip pitch) | 0.76 / 0.50; 0.67 / 0.90 |

Re-sized instead (the sizing loop's view, `REFINE.md` comparison table): at 60 cm the loop changes no drive in the
material cases and the governing knee speed stays 0.95; at 75 cm it spends the saving on a lighter, weaker thigh
twist (back to the AK45-36), so the governing ratio stays high but moves (thigh twist 0.96–0.98, or knee speed 0.93
with 2 mm carbon plates and thin walls). The 60 cm tether-only body takes a smaller MyActuator X4-P12-10 knee (0.91
sustained) and ends 0.69 kg lighter; the 75 cm tether-only body takes a heavier RobStride 06 knee and ends 0.21 kg
*heavier* than its reference.

## 3. Structure

| Material | Density / modulus | Allowable (design) | Minimum practical thickness | Effect here | Notes |
|---|---|---|---|---|---|
| 6061-T6 (current) | 2700 kg/m³ / 68.9 GPa | 70 MPa (25 % of yield, 72 % of fatigue strength) | 3 mm floor in the model (laser-cut stock from 1.0 mm) | reference | cheap, 2–4 day laser cut, takes threads and H7 bearing fits, bends visibly in a fall, conducts motor heat (167 W/m·K) |
| 6061-T6 at 2.54 mm (0.100 in) for the non-bracket plates | as above | as above | 2.54 mm; 0.61× the rigidity of 3 mm, close to 3 mm carbon's 0.67× | −0.18 / −0.16 kg, cost unchanged | the cheapest way to most of carbon's saving; brackets stay 3 mm |
| 7075-T6 | 2810 / 71.7 GPa | 115–125 MPa | 3.175 mm (outsourced stock) | no mass saving where the floor or stiffness governs; about +10 % on 1/8 in stock (brackets: +27 g at 60 cm, +42 g at 75 cm) | 1.8× the yield of 6061: for brackets that must survive falls |
| Carbon/epoxy, quasi-isotropic 3 mm (T700) | 1520 / 45.9 GPa | 95 MPa (estimate: open hole × fatigue and impact) | 3 mm (1.5 fittings factor for bonded inserts); equal rigidity with 3 mm 6061 needs 3.43 mm (3.99 mm stock) | −0.39 / −0.44 kg at 3 mm; −0.20 / −0.22 kg at 3.99 mm | brittle, delaminates without visible marks (tap-test after falls), no threads or bearing fits; about $240/kg raw; the reference plate is on backorder (at least 4 weeks) and the 2–4 day waterjet service stocks 2×2 twill, not quasi-isotropic plate |
| Carbon/epoxy, 0/90 twill | 1450–1520 / 60–64 GPa on axis, shear modulus only 4.1 GPa | 110 MPa on axis | 3 mm | for the stiffness-governed 75 cm tibia | must be cut on the fibre axis |
| Carbon roll-wrapped tubes, bonded 6061 inserts | about 1600 / 60 GPa | 120 MPa | 1.0–2.0 mm wall | the 75 cm spine 2.5× lighter (research estimate, not simulated) | crush at clamps, bonded joints need care and galvanic isolation |
| Ti-6Al-4V | 4430 / 114 GPa | 220 MPa | 1.0 mm sheet | none on plates (same stiffness per mass as aluminium) | pins, axles and fasteners in carbon joints only |
| Magnesium AZ31B | 1770 / 45 GPa | 50 MPa | 3.175 mm | about −25 % at the floor | corrosion and galvanic problems, few outsourced cutters: not recommended |
| Printed fibre-filled nylon (Windform XT 2.0, Nylon 12CF, Onyx) | 1100–1300 / 1.4–9 GPa | 5–12 MPa (estimate) | — | far too flexible for load paths | secondary brackets, guides, mounts |

Sources: Rock West Composites 204-11 and 404-11 plate data (rockwestcomposites.com), DragonPlate price lists,
SendCutSend material and thickness lists (sendcutsend.com/materials), ASM/MatWeb 6061-T6 and 7075-T6 data,
Aerospace Metals Ti-6Al-4V data, Easy Composites tube data, Windform XT 2.0 data sheet; all accessed 2026-11-08.

**Recommendation.** Keep 6061-T6 for prototype 1, with the non-bracket plates at 2.54 mm if the bench test of a
leg confirms their stiffness; optionally 7075-T6 for the hip and thigh brackets (fall robustness, about 10 % more
bracket mass, 30 % more cost). The research's reviewer also estimates that at 75 cm a deeper spine plate (40 mm
instead of 30.4 mm) and a wider tibia plate (30 mm instead of 24 mm) would take off another 0.17 kg in aluminium
(estimate, not simulated). For the long-term robot, plan a hybrid frame: quasi-isotropic
carbon for flat plates, 0/90 carbon for long on-axis members, carbon tubes for the spine and neck spar, and
aluminium for every motor mount, bearing housing and threaded part; titanium fasteners in carbon joints.

## 4. Cosmetic coverings

| Option | Areal mass (with supports) | Effect (research estimate on `60-N-H-10` / `75-M-T-10-bighead`) | Cost and lead time | Durability |
|---|---|---|---|---|
| MJF PA12, 1.6–1.8 mm (current) | 1.94–2.24 kg/m² | reference (shells + supports 1.42 / 2.22 kg on those bodies; 1.32 / 2.20 kg on the refined leaders) | covers line $1.3k / $1.8k ($1.2k / $1.8k on the refined leaders); 3 days | notch-sensitive (notched Izod 3.5 kJ/m²); thin edges and insert bosses crack in falls |
| MJF PA12, graded 1.4 / 1.2 / 1.0 mm | — | −0.26 / −0.38 kg | −$234 / −$321 | 40–60 % of the knock tolerance of 1.6 mm; warp risk on the largest panels |
| **MJF PA11, graded 1.4 / 1.2 / 1.0 mm (recommended)** | — | −0.22 / −0.31 kg | +$81 / +$130; 3 days | elongation 50 % (PA12 20 %), notched Izod 5 kJ/m²: dents and flexes instead of cracking |
| SLS PA12 (PA 2200), graded | — | −0.35 / −0.53 kg | about the same as MJF (estimate) | 4 % strain at break across layers: orient panels |
| Carbon-filled SLS (Windform XT 2.0), 1.0–1.1 mm | — | −0.39 / −0.73 kg | 2–4× PA12 (estimate, not sourced) | brittle (3.8 % elongation) |
| Moulded 2-ply carbon or carbon/aramid skins, 0.44–0.56 mm | 0.76–0.88 kg/m² | −0.61 / −1.38 kg on the large panels | $2.4–7.1k / $4.8–14.2k with moulds; 3–8 weeks | brittle at 1.5 % strain unless aramid plies in knock zones |
| Vacuum-formed ABS | about 1.33 kg/m² | −0.55 / −0.91 kg | $1.5–4k, 20–30 bucks; 1–3 weeks | tough, but split seams and no undercuts |
| EVA foam, 5 mm, sealed and painted | about 0.69 kg/m² | −0.97 / −1.54 kg | $1.6–6k of hand work | the most knock-tolerant; appearance depends on the maker |
| Stretch fabric over a light frame | about 0.59 kg/m² | −1.03 / −1.64 kg | $0.8–2.2k | sags, snags; reads as a costume |

Sources: HP 3D HR PA 12 and PA 11 data sheets, EOS PA 2200 data sheet, Windform XT 2.0 data sheet, Xometry PA11
vs PA12 comparison, HLH Rapid MJF design guide, Protolabs lead times, JLC3DP pricing, Rock West prepreg data,
Formlabs vacuum-forming guide, The Foamory EVA data; all accessed 2026-11-08.

The effects in this table are the research's estimates, computed on two round-two bodies (`60-N-H-10` and the
larger-skull `75-M-T-10-bighead`, which has more cover area) for graded walls with the internal supports raised to
25 % (60 cm) and 29 % (75 cm); compare options with them, not with §2. On the two refined leaders the simulated
lever in §2 thins every wall uniformly to ×0.75 with the supports scaling with the walls: PA11 then saves 0.31 /
0.52 kg for +$20–60, or about 0.26 / 0.44 kg if the supports must rise to 25 / 29 % (the research's reviewer).
Thinner walls keep about 44 % of today's bending stiffness, so add ribs every 50–80 mm (estimate).

**Recommendation.** MJF PA11 with graded walls on every printed panel; EVA foam bumpers at strike points (snout,
jaw tip, tail tip, shins); stretch fabric only over moving gaps (neck-to-trunk, hip and knee gaiters). Moulded
carbon skins on the 75 cm skull, jaw, neck and tail only if the neck servo needs relief (estimated on the
larger-skull body at about −0.46 kg and −30 % of the neck payload for $1.4–4.1k); with the targeted W270 neck servo
at 0.67 it does not.

## 5. Foot pads and grip

The carpet and turning screens assume the carpet band's lower edge, μ 0.54 (assumption A16), from a lightly loaded
shoe on an inclined plane. The recommended bodies' pads carry about 72 kPa (60 cm) and 41 kPa (75 cm, with the larger
pads; 62 kPa without them) in static single support.

| Compound | Expected static μ (estimates from the sources) | Wear (DIN abrasion) | Notes |
|---|---|---|---|
| **Cast polyether PU, 50–60 Shore A tread on a 70–80 A core, 1.0–1.5 mm round studs (recommended)** | carpet 0.70–0.85; wood 0.8–0.9; ceramic tile 0.9–1.0; polished concrete 0.6–0.85 dry; laminate and vinyl 0.42–0.8 | 23–40 mm³ | non-marking; cast in printed moulds or printed as TPU 85 A for a fast trial |
| Natural gum or SBR, 40 Shore A | similar grip | 80–120 mm³ | ages under ozone and UV; aged rubber becomes slippery |
| Silicone, 30–40 Shore A | about 1.0 on clean smooth floors; unknown on carpet | poor (tear 10 N/mm at 40 Shore A) | picks up dust and lint |
| Neoprene | no grip advantage | up to 400 mm³ | backing or bumpers only |
| Hard pads (90 A and harder) | carpet 0.35–0.45 | — | keep hard parts (95 A TPU, printed toes and claws) off the contact path |

What it means for the recommended bodies (ROUND2_REVIEW §6; friction need with the stance foot's yaw moment, from
their turning screens: nominal walk / 0.35 rad/s walking turn / brisk walk), with the bodies they replaced as the
lead for comparison:

| Body | Need | Carpet (0.70–0.85) | Wood (0.8–0.9), tile (0.9–1.0) | Polished concrete (0.6–0.85) | Vinyl (about 0.50), laminate (about 0.42) |
|---|---|---|---|---|---|
| 60 cm narrow carpet-ready (picks 1 and 3) | 0.47–0.48 / 0.53–0.55 / 0.73–0.75 | walk and turn; brisk above the bottom of the range | all | walk and turn; brisk not assured | vinyl: the nominal walk only, barely; laminate: none |
| 75 cm targeted with the bigger feet (pick 2) | 0.36 / 0.41 / 0.56 | all | all | all | vinyl: walk and turn; laminate: walk, turns barely; brisk on neither |
| 60 cm carpet-ready, base width | 0.54–0.55 / 0.62–0.63 / 0.83–0.86 | walk and turn; brisk only at the top of the range | all on tile; brisk marginal on wood | walk; turn marginal; brisk not assured | not even the nominal walk |
| 75 cm targeted, round-two feet | 0.39–0.40 / 0.45–0.46 / 0.61–0.63 | all | all | walk and turn; brisk marginal | vinyl: walk and turn; laminate: the walk only, barely; brisk on neither |

So the controller needs floor-dependent speed and yaw-rate limits (and the tail can cancel some of the stance
foot's yaw), and the 60 cm robot may be confined to carpet, wood, tile and polished concrete. The foot's geometry
moves the need more than the compound can: the narrow stance needs 12–14 % less than the base width (the reason the
60 cm picks use it), longer toes (×1.15) about 8 % less, and larger pads (×1.25) leave it unchanged but raise the
one-foot stop to 1.33 m/s² (ROUND2_REVIEW §7.4–7.6, `round2/refine/REFINE.md`). Before relying on any
number, run inclined-plane and twist tests of the chosen pad at the robot's pressure on the owner's floors, dry and
dusty, before and after two hours of walking (about one day and $100, estimate).

Sources: Hirai and Gunji 2001 (J. Text. Eng. 47(2)); Brandão et al. 2017 (arXiv 1706.08685); Budinski 2018 (STLE
annual meeting); Manoonpong et al. 2016 (PMC5180203); Moore et al. 2011; polyurethane, neoprene and silicone
sheet data sheets; QIDI TPU friction note; Engineering ToolBox friction table; all accessed 2026-11-08.

## 6. Power: battery, tether or both

**What the model says for the demonstrator (60 and 75 cm).**

* The selected pack is a 6S 3.3 Ah LiPo (73 Wh, 0.506 kg; with its cassette 0.68 kg) at the tail base. Walking draws
  71 W (60 cm) and 88 W (75 cm) on the nominal walk, so the pack gives 46.5 and 37.6 min at 75 % usable energy,
  about twice P4's 15–20 min.
* **The pack is also the counterweight.** Round one moved it from the belly to the tail base because that cut
  the 60 cm single-support hip-pitch load from 8.0 to 4.5 N·m (F2). Removing it undoes part of that: the centre of
  mass moves from 34 to 49 mm ahead of the hip at 60 cm and from 64 to 77 mm at 75 cm, the hip-pitch hold rises
  from 3.11 to 4.27 N·m (+37 %) and from 8.18 to 9.35 N·m (+14 %), and the walks change: with the same drives the
  knee goes past its speed limit (1.07 at 60 cm, 1.03 at 75 cm). A tether-only body therefore needs the
  counterweight back (ballast, or the power electronics moved to the tail base), which takes back most of the
  0.6 kg saving. Re-sized without it, the 60 cm body is 0.69 kg lighter on a smaller knee drive and the 75 cm
  body is 0.21 kg *heavier* on a larger one.
* **A tether also loads the robot.** A cable drags behind a walking robot, pulls on the body as it turns and is
  a trip and snag hazard. At 24 V the demonstrator draws 3–4 A on average; the per-drive peaks of the nominal
  walk sum to 129 W (60 cm) and 187 W (75 cm), about 5.4–7.8 A (6.2–8.7 A with the auxiliaries). Brief peaks in stops,
  pushes and falls will be higher.

**Recommendation for the demonstrator: battery plus an external DC input.** Keep the removable 6S pack in the
tail-base cassette on every run, and add an external 24 V input at the tail base through a priority power path
(an LTC4417-class controller: three inputs, 2.5–36 V, blocks reverse and cross conduction). P4 already asks for both
("removable battery for 15–20 min, external-power operation"), and the cost model already carries a bench and
tether supply line. The research's design:

* The priority path connects one source at a time (the tether when present, after 256 ms inside its voltage
  window), so the battery rides through a tether drop-out but does not share the peaks: the tether carries them,
  about 6.2 A at 60 cm and 8.7 A at 75 cm (summed drive peaks plus auxiliaries), against a mean of 2.9 / 3.7 A.
  5 m of SJOOW 16/2 (82 g/m, rated 13 A) drops 0.82 / 1.14 V at those peaks. If real load sharing is wanted, use an
  ideal-diode OR (LTC4359 class) with the supply current-limited near the mean. A Mean Well RSP-750-24 class supply
  (31 A, constant-current overload with auto-recovery) feeds it.
* The tether input's over-voltage trip and an onboard regeneration clamp at 27 V or below: the 60 cm CubeMars AK45
  drives accept only 15–28 V, and the supply's own over-voltage trip sits at 27.6–32.4 V. The power path stops the
  LiPo from being charged from the tether.
* A Category 1 E-stop (controlled stop, then power off) that cuts the tether and the battery together, by switching
  off the power-path FETs plus a manual disconnect, or with an economised or latching contactor. A plain TE LEV100
  contactor's 6 W coil would itself cost about 3.6 min of runtime at 60 cm.
* **Hang the tether from an overhead boom; never trail it.** Five metres of cable on the floor drags 1.2–3.6 N,
  against a single-support tip threshold of about 7.1 N at 60 cm and 9.8 N at 75 cm for a pull at the tail-base
  cable exit, and 2 N of drag raises the 60 cm brisk walk's friction need from 0.86 to 0.90–0.95. No practical
  breakaway connector releases below the tip threshold (an Anderson Powerpole pair separates at 27–44 N). The
  balancer must carry less than the hanging cable (rated under 0.5 kg, or a counterweighted line): a stronger one
  lifts the tail base and adds nose-down hip-pitch load.
* For battery-free runs, an equal-mass dummy puck (0.678 kg) in the cassette keeps the centre of mass and the
  controller's model unchanged; `BEHAVIOR_SPEC.md` §4 already requires the cassette or an equal-mass dummy on
  tethered runs.
* Estimated added cost: about $0.2–0.4k for the input, power path, clamp and puck, plus $0.3–1.5k for a boom if the
  bench lot's catch frame cannot carry it (estimates).

**A lighter pack is the one weight lever in the power system.** A 6S1P Li-ion pack of Molicel P45B cells
(434 g, 97–100 Wh, $60, steel-can cells) weighs 72 g less and still counts as under 100 Wh. Its runtime depends on
how much of its sloped discharge curve is usable above the 21 V bus floor: about 54 / 43 min at an estimated 65 %
usable (62 / 50 min at the LiPo's 75 %), against 46.5 / 37.6 min. It is a different shape (about 65 × 43 × 71 mm
against 138 × 42 × 43 mm, 28 mm taller), so the cassette and its removal path must be redesigned and re-screened;
not simulated here. Hobby packs of this kind have no battery-management system.

**The 3–4 ft robot (100–120 cm, P1).** Round one sized these bodies with the Jetson's compute budget. With the
Pi 5's budget (9.5 W, plus the same 4 W MCU and link and the sensors, through the 0.88 converter) their walking
draw is about 130–135 W at 100 cm and 180–190 W at 120 cm (round-one records `100-M-T-10`, `100-M-H-10`,
`120-M-T-10`, `120-M-H-10`, adjusted; round one's walk-accuracy fault changed continuous loads by less than 2 %,
F14):

| Body (mass) | Bus | Pack | Pack mass (share of the robot) | Walking runtime at 75 % usable |
|---|---|---|---|---|
| `100-M-T-10` (28.4 kg, 130 W) | 24 V | 6S 5 Ah LiPo, 111 Wh | 0.76 kg (2.7 %) | about 38 min |
| | | 6S2P Li-ion (Molicel P45B), 200 Wh | 0.86 kg (3.0 %) | about 69 min |
| `120-M-T-10` (43.6 kg, 180 W) | 24 V | 6S 5 Ah LiPo | 0.76 kg (1.7 %) | about 28 min |
| | | 6S2P Li-ion, 200 Wh | 0.86 kg (2.0 %) | about 50 min |
| `120-M-H-10` (43.1 kg, 190 W) | 48 V | 12S2P Li-ion, 400 Wh | 1.72 kg (4.0 %) | about 94 min |

So **yes, run the 3–4 ft robot on its battery.** About an hour of walking costs 2–4 % of its mass in packs (2.6–5.2 %
with a cassette), against 3.5–4.7 % for the demonstrator's pack (4.6–6.3 % with its cassette). What changes at that size is not mass but:

* **Falls and pack protection.** A fall from standing carries about 160 J at 100 cm and 300 J at 120 cm (mass ×
  g × centre-of-mass height), against 36 J and 63 J for the demonstrators. The pack needs a protected cassette
  inside the pelvis or tail-root frame and a battery-management system with temperature telemetry.
* **Bus voltage.** At 100–120 cm the research recommends 48 V: half the current of 24 V, and a 48 V tether stays
  within the safety extra-low-voltage limit (60 V DC). It proposes about 390 Wh as two hot-swappable 12S strings,
  each two 6S1P Li-ion modules of 97 Wh, about 1.7–2.0 kg in all (3–7 % of mass) for about 51–117 min at 345–150 W.
  One string can be swapped while the other carries the robot. Comparable robots carry more: Boston Dynamics Spot
  carries a 605 Wh, 4.2 kg battery on 32.5 kg (13 % of mass) for about 90 min.
* **Travel and shipping.** Spare packs of 100 Wh or less go in carry-on luggage without approval; 101–160 Wh
  needs the airline's approval (two spares at most); above 160 Wh is forbidden on passenger aircraft. Lithium-ion
  packs over 100 Wh ship as fully regulated Class 9 dangerous goods, and from January 2026 IATA requires lithium-ion
  cells and batteries above 2.7 Wh, sub-100 Wh packs included, to ship at 30 % state of charge or less. Sub-100 Wh
  modules avoid the over-100 Wh rules and the airline approval, not the 30 % limit. Every transported pack needs
  UN 38.3 test documents.
* **The tether's place.** A 28–56 kg robot needs a fall-arrest gantry during development anyway; the tether belongs
  on that gantry and on a charging dock, not on the floor. Its job (navigating, following people, voice commands)
  cannot be done on a cable.

Sources for this section: the round-two and refinement records (`round2/refine/carpet_60`, `targeted_75`,
`mat60_tether`, `mat75_tether`, `fixed_drives.json`); round one's records (`round1/stage_a/candidates/100-*.json`,
`120-*.json`); Analog Devices LTC4417 and LTC4359 data; Mean Well RSP-750 specification; AutomationDirect SJOOW
cable data; IEC 60228 conductor resistance; Anderson Power DS-PP45 data; LEMO breakaway-connector white paper; Boston
Dynamics Spot specification sheet; Unitree G1 and H1 specifications; Battery University BU-205 and BU-409; CSATF,
TSA and FAA lithium-battery travel rules; the 2025 IATA battery guide summary (The Compliance Center); Defiance RC
and akkuteile.de P45B pack and cell data; accessed 2026-11-08.

