# Anatomical references and how the template uses them

The full sourced record is `designlab/data/research/anatomy_2026-11-07.json`: 37 references,
238 measurements and 39 derived ratios, each with source, table or page, and licence, accessed
2026-11-07. The skeletal template (`designlab/template.py`) is expressed in units of hip height `H`.
Each value below is classed as one of:

* **Reference**: within a published or derived range.
* **Adaptation**: changed for engineering reasons, with the reason stated.
* **Interpretation**: an artistic choice where the science leaves room, or where the robot needs a decision.

| Template quantity | Robot (round one) | Published / derived | Main sources | Class |
|---|---|---|---|---|
| Skull length / hip height | 0.47 | 0.44 (0.39–0.48) | H2011 tbl 3; GE2017 tbl 1; HS2003 | Reference |
| Skull length / femur | 1.06 | 1.04 (0.92–1.10) | P2020 tbl 1; PC2016 | Reference |
| Skull width at jaw joints / length | 0.55 (T), 0.58 (H), 0.62 (S) | 0.64 (0.50–0.71) | GE2017; PAD2022; HS2003 ("extremely broad posteriorly") | Reference |
| Skull depth / length (without mandible) | ≈ 0.39 | 0.41 (0.32–0.435) | CARR2020 S1; HS2003 | Reference |
| Gleno-acetabular distance | 0.88 H | 0.211–0.225 of body length ≈ 0.88 H | H2011 tbl 3 (SUE, Stan, CM 9380) | Reference |
| Femur : tibia : metatarsal III | 1 : 0.90 : 0.53 | 1 : 0.90 (0.79–0.93) : 0.53 (0.52–0.60) | PC2016 tbl 1; P2020 tbl 1 | Reference |
| Standing limb posture | Femur 22° forward, tibia 28° back, metatarsus 15° forward; hip height = 0.92 × chain | "Neither very columnar nor very flexed"; hip ≈ 0.82–0.94 × leg length | G2009; H2005; HG2002 | Reference |
| Hock (ankle) | Fixed raised hock; the toes carry the load | Digitigrade: the metatarsus does not touch the ground | — | Adaptation: fused tibia–metatarsus, so the actuated ankle sits at the MTP joint |
| Foot length / hip height; toe splay | 0.26; ±33° (66° total) | 0.25 (0.20–0.32); 70° total (51–80) | MC2014; F2026 | Reference |
| Tail length (first free caudal → tip) | 1.55 H (default), 2.29 H (`anat` variant) | 0.51–0.59 of body length ≈ 2.3 H | H2011 tbl 3 | Adaptation: shortened by default to limit turning sweep, floor footprint and tail inertia. The `anat` variant restores it |
| Chest half-width (fleshed) | 0.22 H (S), 0.20 H (H), 0.18 H (T) | Ribcage ≥ 0.086–0.091 of body length; +17.8 % soft tissue outside the ribs → ≈ 0.23 H | PAD2022; Field Museum 2018 gastralia; B2009 fleshing rule | Reference (S); Interpretation (T leaner by design) |
| Belly line, pubic keel | Gastral basket deep at the chest, slightly concave behind it, distinct pubic boot between the thighs | "Very barrel-chested"; gastralia added in the 2018 SUE remount | FM_MAKEOVER; BR2003 | Interpretation, constrained by reference |
| Thigh | Muscular drumstick fairing, broad at the hip, tapering to a slender knee | Large thigh and caudofemoral muscles; femur projects ventrolaterally | PC2011; BR2003; SN2019 | Interpretation |
| Arms | Fixed, low and forward on the chest | 2018 remount lowered the arms and brought them forward | FM_MAKEOVER; PAD2022 | Reference |
| Head carriage (level posture) | Head and neck centre of mass 1.29 H ahead of and 0.18–0.19 H above the hip at 0° (1.27 H / 0.34–0.36 H at 10°; 1.21 H / 0.50 H at 20°) | B2009 digital mounts: ≈ 1.23 H ahead, +0.07–0.13 H above. PAD2022 MOR 555 S-neck mount: higher | B2009 tbl 2–3; PAD2022 | Interpretation (between mounts; also varied by posture) |
| Neck | Two-segment S approximation, 0.32 B along the curve | 1.5 × skull length along a three-segment S (one cast mount) | PAD2022 | Interpretation: shorter, which keeps the head closer to the hips (hip-pitch load) |
| Centre of mass ahead of hip | 0.12–0.15 H (battery in the tail base; 60–120 cm reference robots); 0.17–0.18 H (belly battery) | 0.30–0.61 femur lengths ≈ 0.13–0.27 H | H2011; B2009 | Adaptation: the robot's tail is light (3–5 % of mass vs 10–21 % in the animal), so the battery moves to the tail base to restore balance. The anatomical-length tail variant moves the centre of mass back to 0.09–0.11 H |
| Mass fractions | Head+neck 5.0–5.5 % (includes the head-mounted camera), tail 2.9–4.5 %, one leg 23.5–26 % | Head 4–11 %, tail 10–21 %, one leg 9.5–16 % | H2011 tbl 4–8; B2009 | Hardware-driven: actuators make robot legs heavy, and shells make head and tail light |
| Foot spacing (track gauge) | 0.24–0.49 H (narrow to broad variants, all sizes; balanced 0.31–0.40 H) | Left-right track centroid spacing ≈ 0.07 H (0.046–0.09); pace angulation 164–171° | F2026; MC2014 | **Adaptation, with a large gap.** The animal walked nearly single-file. Robot feet are limited by hip hardware width and swing clearance (`SCREENING.md`) |
| Walking gait | Reference walk Fr 0.012, stride 0.55 H, ~2 steps/s at 60 cm | Preferred walk Fr ≈ 0.05 (0.02–0.13), relative stride 1.0–1.5 H | VB2021; F2026; MC2014 | Adaptation for round-one sizing. The behavior spec proposes moving toward Fr 0.02–0.05 with longer strides |

## Independent replicate

A second, independent research pass (`anatomy_2026-11-07_replicate.json`: 34 references, 45 ratios) agrees on every calibrated value:
* skull/hip height 0.47 (0.40–0.57);
* rear skull width/length 0.63;
* tibia/femur 0.91 and MT III/femur 0.536;
* gleno-acetabular distance/body length 0.212;
* tail/body length 0.55.

It adds four points:
* **Chest.** Fleshed chest width/hip height is 0.43 (0.39–0.57) and chest width/height is 0.92 (0.72–0.98). The animal was barrel-chested; our S chest is deeper than it is wide (≈ 0.75), and T sits just below the low end of the width range by design.
* **Head position.** The snout tip lies 1.13–1.5 skull lengths ahead of the glenoid. On the round-one models it is 1.59 skull lengths ahead of the shoulder reference point, so the robot's head is carried slightly further forward than in mounts. A steeper neck is a round-two option, and it would also lower hip-pitch load.
* **Glenoid height.** The glenoid sits ≈ 0.16 H below the acetabulum (see the torso-inclination definition).
* **Walking gait.** Walking Froude 0.05–0.25 and relative stride 1.2–1.5 H.

## What this means for the design

The robot's proportions follow the animal where hardware allows: skull, trunk length, leg ratios,
foot, arms and chest volume. They depart where the hardware decides:
* the legs are heavy (23–26 % of mass each) and the tail is light (3–5 %), so the centre of mass sits near the hip only because the battery moves into the tail base;
* the feet sit about 5× wider apart than tyrannosaurid tracks;
* the default tail is shorter (a stated option, not a constraint).

The most important open visual decisions are posture/head carriage and tail length. Round one
presents both as explicit choices.

## Licensing

Measurements and derived ratios are factual and cited. No figure, photograph, skeletal drawing,
scan or traced silhouette from any source is reproduced. Every image in this study is rendered from the generated
models. CC-BY sources (H2011, PC2016, GE2017, PAD2022, CARR2020, B2009, SN2019, S2017, VB2021) may
be quoted with attribution. MC2014 is CC0. Field Museum pages and closed-access papers (BR2003,
HG2002, P2020 via Plazi) are cited for facts only. Full citations, DOIs, URLs and per-source
licence notes are in the JSON record. The research agent also recorded open questions there:
SUE's exact skull length and MT III, unit inconsistencies in one table, and three senses of
"hip height".
