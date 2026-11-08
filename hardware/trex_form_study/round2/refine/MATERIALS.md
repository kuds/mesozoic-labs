# Materials and power (round two, overnight)

Owner questions (overnight, 7–8 Nov): which materials for the structure and the cosmetic coverings, carbon fibre
or other light, strong materials; anything on the margin worth adjusting; a non-battery (direct power) version;
battery operation on the 3–4 ft robot.

Method. Sourced research on the materials, coverings, foot pads and power architecture (cited inline; accessed
2026-11-08; the research workflow's raw output is in the session scratch). Simulation of each lever on the two
refined leaders, `60-M-T-10` carpet-ready (`round2/refine/carpet_60`) and `75-M-T-10` with the targeted upgrade
(`round2/refine/targeted_75`), in two ways (`designlab/refine.py`):

* **re-sized** (`round2/refine/mat60_*`, `mat75_*`): the normal sizing loop picks drives and pack for the lighter
  body;
* **same drives** (`round2/refine/fixed_drives.json`, `REFINE.md` last table): the lighter body keeps the
  reference body's drives and pack, so the saving shows up as margin.

Every number is a model estimate. None is measured.

## Short answers

1. **Lighter materials do not move the limit that binds the leading bodies.** On both refined leaders the
   governing limit is the knee's *speed* in the Froude 0.05 walk (0.95). That is set by the walk's geometry and
   timing, not by mass. Every material case leaves it at 0.94–0.95. Lighter bodies lower the torques, the
   walking power and the neck and tail loads, so they buy torque headroom and runtime, not walking-speed
   headroom. Runtime already exceeds the 15–20 min requirement by about 2×.
2. **Carbon-fibre frame plates: not for prototype 1.** With the sourced 3 mm quasi-isotropic plate they save
   0.39 kg (60 cm) or 0.44 kg (75 cm) for $1.6k or $2.1k more (nominal), about $4–5k per kilogram, and they take
   the 75 cm body to $20.5k nominal, over P4's $20k. Carbon cannot hold bearing fits or threads, delaminates out
   of sight in a fall and needs bonded aluminium inserts. Keep 6061-T6. A hybrid frame (carbon plates and tube
   spars, aluminium wherever a motor, bearing or thread attaches) is the right direction for the long-term
   robot, where stiffness rather than the 3 mm minimum sets the plates.
3. **Coverings: the best value lever.** Walls about a quarter thinner save 0.31–0.35 kg (60 cm) or 0.52–0.59 kg
   (75 cm) at no extra cost. Most of the gain is on the head and tail, where mass sits on a lever arm. The research
   recommends MJF **PA11** for the thinner walls instead of today's PA12: 2.5× the elongation and 1.4× the notched
   impact strength, so walls dent rather than crack, for about 30 % more per volume, which the thinner walls pay
   for. With PA11 at ×0.75 the neck servo falls from 0.76 to 0.70 of its rating (60 cm) and from 0.67 to 0.60
   (75 cm), and the 75 cm tail servo's peak from 0.89 to 0.77; cost changes by $20–60.
4. **Foot pads: the grip problem is a pad-compound problem.** The 60 cm brisk walk needs friction 0.73–0.88 once
   the stance foot's yaw moment is counted (F21). A cast polyether polyurethane tread (50–60 Shore A, with 1.0–1.5
   mm studs) is expected to give about 0.70–0.85 on carpet and 0.8–1.0 on wood and tile, but laminate and vinyl
   can be as low as 0.42–0.50. Measure the chosen pad on the owner's floors before relying on it.
5. **Direct power (tether-only): no.** The battery is also the counterweight at the tail base (F2). Without it the
   centre of mass moves 13–15 mm forward, the hip-pitch hold rises 14–37 %, and with the same drives the knee goes
   past its speed limit (1.03–1.07). Keep the removable battery and add an external DC input, as P4 already asks
   ("removable battery … external-power operation"); use the tether for bench work.
6. **3–4 ft robot: battery operation is easy in mass terms.** At 100–120 cm a 0.9–1.7 kg pack gives about 50–95
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

Only structure and coverings (28 % and 32 %) change with material. Most 60 cm plates sit at the 3 mm minimum
thickness, so their mass is density × minimum thickness; at 75 cm the tibia and spine are stiffness-governed.

## 2. Simulated levers

Same drives as the reference body (the margin view). Δ is against the reference.

| Lever | 60 cm Δ mass | 60 cm: walking power / runtime | 60 cm cost Δ (nominal / high) | 75 cm Δ mass | 75 cm: walking power / runtime | 75 cm cost Δ | Knee speed (60 / 75) | Highest leg torque (60 / 75) | Neck / tail peak (60; 75) |
|---|---|---|---|---|---|---|---|---|---|
| Reference | 10.83 kg | 70.8 W / 46.5 min | $14.8k / $20.6k | 14.58 kg | 87.7 W / 37.6 min | $18.4k / $25.4k | 0.95 / 0.95 | 0.76 / 0.78 | 0.76 / 0.52; 0.67 / 0.89 |
| Carbon plates, sourced 3 mm quasi-isotropic | −0.39 kg | 69.2 W / 47.6 min | +$1.6k / +$2.3k | −0.44 kg | 85.5 W / 38.6 min | +$2.1k / +$3.1k | 0.95 / 0.95 | 0.74 / 0.75 | 0.75 / 0.51; 0.66 / 0.89 |
| Carbon plates, 2 mm floor (optimistic) | −0.62 kg | 68.3 W / 48.2 min | +$1.6k / +$2.3k | −0.81 kg | 83.6 W / 39.4 min | +$2.1k / +$3.1k | 0.95 / 0.95 | 0.72 / 0.73 | 0.75 / 0.51; 0.65 / 0.89 |
| Covering walls ×0.75 (PA12) | −0.35 kg | 68.8 W / 47.9 min | −$0.3k / −$0.5k | −0.59 kg | 84.0 W / 39.3 min | −$0.4k / −$0.7k | 0.95 / 0.94 | 0.74 / 0.75 | 0.69 / 0.44; 0.59 / 0.76 |
| **PA11 coverings, walls ×0.75 (recommended)** | −0.31 kg | 69.0 W / 47.7 min | +$20 / +$40 | −0.52 kg | 84.4 W / 39.1 min | +$30 / +$60 | 0.95 / 0.94 | 0.74 / 0.75 | 0.70 / 0.45; 0.60 / 0.77 |
| Sourced carbon + walls ×0.75 | −0.74 kg | 67.3 W / 49.0 min | +$1.3k / +$1.8k | −1.02 kg | 81.9 W / 40.3 min | +$1.7k / +$2.4k | 0.95 / 0.94 | 0.71 / 0.72 | 0.68 / 0.43; 0.59 / 0.75 |
| Sourced carbon + PA11 walls ×0.75 | −0.70 kg | 67.5 W / 48.8 min | +$1.6k / +$2.3k | −0.95 kg | 82.3 W / 40.1 min | +$2.2k / +$3.2k | 0.95 / 0.94 | 0.71 / 0.73 | 0.69 / 0.44; 0.60 / 0.77 |
| Tether-only (no pack or cassette; 0.12 kg connector) | −0.61 kg | 70.4 W / tethered | −$0.1k / −$0.1k | −0.59 kg | 87.9 W / tethered | −$0.1k / −$0.0k | **1.07 / 1.03** | 0.72 / 0.82 (hip pitch) | 0.76 / 0.50; 0.67 / 0.90 |

Re-sized instead (the sizing loop's view, `REFINE.md` comparison table): the lighter bodies keep the same governing
ratios, because the loop spends the saving on lighter drives. With 2 mm carbon plates the 75 cm thigh twist
goes back to the AK45-36 (0.96); the 60 cm tether-only body takes a smaller MyActuator X4-P12-10 knee (0.91
sustained) and ends 0.70 kg lighter; the 75 cm tether-only body takes a heavier RobStride 06 knee and ends 0.21 kg
*heavier* than its reference.

## 3. Structure

| Material | Density / modulus | Allowable (design) | Minimum practical thickness | Effect here | Notes |
|---|---|---|---|---|---|
| 6061-T6 (current) | 2700 kg/m³ / 68.9 GPa | 70 MPa (25 % of yield, 72 % of fatigue strength) | 3 mm floor in the model (laser-cut stock from 1.0 mm) | reference | cheap, 2–4 day laser cut, takes threads and H7 bearing fits, bends visibly in a fall, conducts motor heat (167 W/m·K) |
| 7075-T6 | 2810 / 71.7 GPa | 115–125 MPa | 3.175 mm (outsourced stock) | no mass saving where the floor or stiffness governs; +4–10 % mass | 1.8× the yield of 6061: for brackets that must survive falls |
| Carbon/epoxy, quasi-isotropic 3 mm (T700, waterjet) | 1520 / 45.9 GPa | 95 MPa (estimate: open hole × fatigue and impact) | 3 mm (1.5 fittings factor for bonded inserts) | −0.39 kg / −0.44 kg | brittle, delaminates without visible marks (tap-test after falls), no threads or bearing fits; about $240/kg raw |
| Carbon/epoxy, 0/90 twill | 1450–1520 / 60–64 GPa on axis, shear modulus only 4.1 GPa | 110 MPa on axis | 3 mm | for the stiffness-governed 75 cm tibia | must be cut on the fibre axis |
| Carbon roll-wrapped tubes, bonded 6061 inserts | about 1600 / 60 GPa | 120 MPa | 1.0–2.0 mm wall | the 75 cm spine 2.5× lighter (research estimate, not simulated) | crush at clamps, bonded joints need care and galvanic isolation |
| Ti-6Al-4V | 4430 / 114 GPa | 220 MPa | 1.0 mm sheet | none on plates (same stiffness per mass as aluminium) | pins, axles and fasteners in carbon joints only |
| Magnesium AZ31B | 1770 / 45 GPa | 50 MPa | 3.175 mm | about −25 % at the floor | corrosion and galvanic problems, few outsourced cutters: not recommended |
| Printed fibre-filled nylon (Windform XT 2.0, Nylon 12CF, Onyx) | 1100–1300 / 1.4–9 GPa | 5–12 MPa (estimate) | — | far too flexible for load paths | secondary brackets, guides, mounts |

Sources: Rock West Composites 204-11 and 404-11 plate data (rockwestcomposites.com), DragonPlate price lists,
SendCutSend material and thickness lists (sendcutsend.com/materials), ASM/MatWeb 6061-T6 and 7075-T6 data,
Aerospace Metals Ti-6Al-4V data, Easy Composites tube data, Windform XT 2.0 data sheet; all accessed 2026-11-08.

**Recommendation.** Keep 6061-T6 for prototype 1; optionally 7075-T6 for the hip and thigh brackets (fall
robustness, about 4 % more mass, 30 % more cost). For the long-term robot, plan a hybrid frame: quasi-isotropic
carbon for flat plates, 0/90 carbon for long on-axis members, carbon tubes for the spine and neck spar, and
aluminium for every motor mount, bearing housing and threaded part; titanium fasteners in carbon joints.

## 4. Cosmetic coverings

| Option | Areal mass (with supports) | Effect (60 / 75 cm, research estimate) | Cost and lead time | Durability |
|---|---|---|---|---|
| MJF PA12, 1.6–1.8 mm (current) | 1.94–2.24 kg/m² | reference (shells + supports 1.42 / 2.22 kg) | covers line $1.3k / $1.8k; 3 days | notch-sensitive (notched Izod 3.5 kJ/m²); thin edges and insert bosses crack in falls |
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

**Recommendation.** MJF PA11 with graded walls on every printed panel; EVA foam bumpers at strike points (snout,
jaw tip, tail tip, shins); stretch fabric only over moving gaps (neck-to-trunk, hip and knee gaiters). Moulded
carbon skins only on the 75 cm skull, jaw, neck and tail if the neck servo needs relief (about −0.46 kg and
−30 % of the neck payload for $1.4–4.1k, estimates); with the targeted W270 neck servo at 0.67 it does not.

## 5. Foot pads and grip

The carpet and turning screens assume the carpet band's lower edge, μ 0.54 (assumption A16), from a lightly loaded
shoe on an inclined plane. The robot's pads carry 74 kPa (60 cm) and 65 kPa (75 cm) in single support.

| Compound | Expected static μ (estimates from the sources) | Wear (DIN abrasion) | Notes |
|---|---|---|---|
| **Cast polyether PU, 50–60 Shore A tread on a 70–80 A core, 1.0–1.5 mm round studs (recommended)** | carpet 0.70–0.85; wood 0.8–0.9; ceramic tile 0.9–1.0; polished concrete 0.6–0.85 dry; laminate and vinyl 0.42–0.8 | 23–40 mm³ | non-marking; cast in printed moulds or printed as TPU 85 A for a fast trial |
| Natural gum or SBR, 40 Shore A | similar grip | 80–120 mm³ | ages under ozone and UV; aged rubber becomes slippery |
| Silicone, 30–40 Shore A | about 1.0 on clean smooth floors; unknown on carpet | poor (tear 14 N/mm) | picks up dust and lint |
| Neoprene | no grip advantage | up to 400 mm³ | backing or bumpers only |
| Hard pads (90 A and harder) | carpet 0.35–0.45 | — | keep hard parts (95 A TPU, printed toes and claws) off the contact path |

What it means for the screens: the 75 cm brisk walk (0.51–0.59 with the yaw moment) is covered on every floor
except laminate at its pessimistic edge. The 60 cm brisk walk (0.73–0.88) is not assured on carpet or laminate:
the controller needs floor-dependent speed and yaw-rate limits, or the pads must spread (a 20 % larger pad radius
cuts the yaw term by about 17 %; the `pads_large` experiment tests ×1.25 pads, `REFINE.md`). Before relying on
any number, run inclined-plane and twist tests of the chosen pad at the robot's pressure on the owner's floors,
dry and dusty, before and after two hours of walking (about one day and $100, estimate).

Sources: Hirai and Gunji 2001 (J. Text. Eng. 47(2)); Brandão et al. 2017 (arXiv 1706.08685); Budinski 2018 (STLE
annual meeting); Manoonpong et al. 2016 (PMC5180203); Moore et al. 2011; polyurethane, neoprene and silicone
sheet data sheets; QIDI TPU friction note; Engineering ToolBox friction table; all accessed 2026-11-08.

## 6. Power: battery, tether or both

{{POWER_SECTION}}
